//! The Qwen3.6 MTP head as the lane core's drafter (decode.mtp_round): the kept rows absorbed, then a chain of drafts.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const q = @import("qwen27");
const mtp = @import("mtp.zig");

const dr = lanes.drafter;
const bf = 2;
/// MTP drafts a round at most (cuda.mtp DEPTH).
pub const depth = 3;
/// A chain ends after a draft the head gives less probability than this (cuda.mtp CONFIDENCE).
pub const confidence: f32 = 0.3;

/// A drawn draft and its temperature-1 probability under the head's row.
const Pick = struct { token: u32, prob: f32 };

/// One stream's head state: its cache, and the chain it holds for its next verify.
const Lane = struct {
    cache: mtp.Cache,
    tokens: [depth]u32 = undefined,
    count: usize = 0,
};

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    head: *mtp.Head,
    lanes: std.AutoHashMapUnmanaged(*lanes.Stream, *Lane) = .empty,
    step_ms: f64 = 1.0,
    cand_values: []f32 = &.{},
    cand_ids: []i64 = &.{},
    values: []f64 = &.{},
    row: []u16 = &.{},
    stats_mod: ?cuda.Module = null, // qwen_route.cu's row-stats kernel
    stats_fn: cuda.Function = undefined,

    /// Takes the head; times one chain step for the depth rule.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, head: *mtp.Head) !Drafter {
        var d: Drafter = .{ .gpa = gpa, .head = head };
        errdefer d.deinit();
        d.cand_values = try gpa.alloc(f32, q.state.max_candidates);
        d.cand_ids = try gpa.alloc(i64, q.state.max_candidates);
        d.values = try gpa.alloc(f64, q.state.max_candidates);
        d.row = try gpa.alloc(u16, head.w.ids.len);
        d.stats_mod = try cuda.Module.load(head.e.ctx.d, cuda.kernels.qwen_route);
        d.stats_fn = try d.stats_mod.?.function("tf_qwen_row_stats");
        d.step_ms = try d.time(io);
        return d;
    }

    pub fn deinit(d: *Drafter) void {
        var it = d.lanes.valueIterator();
        while (it.next()) |l| {
            l.*.cache.deinit();
            d.gpa.destroy(l.*);
        }
        d.lanes.deinit(d.gpa);
        d.gpa.free(d.cand_values);
        d.gpa.free(d.cand_ids);
        d.gpa.free(d.values);
        d.gpa.free(d.row);
        if (d.stats_mod) |*m| m.unload();
    }

    pub fn drafter(d: *Drafter) dr.Drafter {
        return .{ .ptr = d, .vtable = &.{ .taps = taps, .facts = facts, .open = open, .absorb = absorb, .hold = hold, .held = held, .held_count = heldCount, .release = release } };
    }

    /// A chain step's cost: one row through the head, the draft head and a draw, median of five.
    fn time(d: *Drafter, io: std.Io) !f64 {
        const g = d.head.e.w.g;
        var cache = try mtp.Cache.init(d.head.e.ctx.d, g, 64);
        defer cache.deinit();
        const f = d.head.e.forward();
        try f.ops.fill32(cache.last, 0, g.hidden * bf / 4);
        var times: [5]f64 = undefined;
        const clock = std.Io.Clock.awake;
        for (0..6) |rep| {
            try f.ops.s.synchronize();
            const t0 = clock.now(io).toNanoseconds();
            try d.head.forward(&cache, cache.last, &.{0}, 0);
            try d.head.scoresOf(d.head.normed);
            _ = try d.draw(null, 1, true);
            if (rep > 0) times[rep - 1] = @as(f64, @floatFromInt(clock.now(io).toNanoseconds() - t0)) / 1e6;
        }
        std.mem.sort(f64, &times, {}, std.sort.asc(f64));
        return times[2];
    }

    fn self(p: *anyopaque) *Drafter {
        return @ptrCast(@alignCast(p));
    }

    const final_taps = [_]u32{q.features.final_tap};

    fn taps(_: *anyopaque) []const u32 {
        return &final_taps;
    }

    fn facts(p: *anyopaque) dr.Facts {
        const d = self(p);
        return .{ .depth = depth, .stops = true, .step_ms = d.step_ms, .stream_bytes = mtp.Cache.bytes(d.head.e.w.g, d.head.e.context) };
    }

    fn open(p: *anyopaque, s: *lanes.Stream) anyerror!void {
        const d = self(p);
        const capacity = @min(d.head.e.context, s.prompt().len + s.max_new + q.state.max_rows);
        const l = try d.gpa.create(Lane);
        errdefer d.gpa.destroy(l);
        l.* = .{ .cache = mtp.Cache.init(d.head.e.ctx.d, d.head.e.w.g, capacity) catch |err| return if (err == error.OutOfMemory) err else error.OutOfDeviceMemory };
        errdefer l.cache.deinit();
        try d.lanes.put(d.gpa, s, l);
    }

    /// One head call over `rows` of the device rows at `states` (token t + 1 after row t), the last row's output kept.
    fn call(d: *Drafter, l: *Lane, states: u64, tokens: []const u32, p0: usize) !void {
        const hid = d.head.e.w.g.hidden;
        try d.head.forward(&l.cache, states, tokens, p0);
        try d.head.e.forward().ops.copy(l.cache.last, d.head.normed + (tokens.len - 1) * hid * bf, hid * bf);
    }

    /// Kept rows into the head: a prompt in the chunks the target's prefill cut it in, a verify's rows in one call.
    fn absorb(p: *anyopaque, items: []const dr.Absorb) anyerror!void {
        const d = self(p);
        const hid = d.head.e.w.g.hidden;
        for (items) |it| {
            const l = d.lanes.get(it.stream) orelse return error.NotDrafting;
            const f = it.features;
            if (f.space != .device or f.dtype != .bf16 or f.row_bytes != hid * bf) return error.UnsupportedFeatures;
            const rows: usize = f.rows;
            if (it.follow.len < rows) return error.ShortFollow;
            const base = f.buffer + f.offset;
            const start: usize = @intCast(it.start);
            const len = it.stream.prompt_len;
            var at: usize = 0;
            if (start + rows == len - 1 and rows > 1) { // the prompt but its last row, as its chunks were cut
                const n = (len - start + q.engine.prompt_rows - 1) / q.engine.prompt_rows;
                for (0..n) |j| {
                    const b = q.engine.chunkBounds(start, len, q.engine.prompt_rows, j);
                    const end = @min(b[1] - 1, start + rows);
                    if (end <= start + at) continue;
                    try d.call(l, base + at * hid * bf, it.follow[at .. end - start], start + at);
                    at = end - start;
                }
            } else {
                try d.call(l, base, it.follow[0..rows], start);
                at = rows;
            }
            l.cache.pos = start + rows;
        }
    }

    /// The head's row maximum and exp-sum into the pick buffer's words 1 and 2 (torch.softmax's float32 normalizer).
    fn rowStats(d: *Drafter) !void {
        const h = d.head;
        var a: cuda.Args = .{};
        a.add(h.logits);
        a.add(@as(c_int, @intCast(h.w.ids.len)));
        a.add(h.pick + 4);
        const f = h.e.forward();
        try cuda.launch.launch(d.stats_fn, .{ .grid = .{ .x = 1, .y = 1, .z = 1 }, .block = .{ .x = 1024 } }, f.ops.s, &a);
    }

    /// Probability of a logit under the row's stats: exp(x - max) / sum, as torch.softmax in float32.
    fn probOf(x: f32, stats: [2]f32) f32 {
        return @exp(x - stats[0]) / stats[1];
    }

    /// The target's draw rule on the head's logits, mapped to a token (decode.draft): the greedy column, or a keyed draw; with its probability if `more` (a chain goes on by it).
    fn draw(d: *Drafter, rule: ?lanes.Sampling, position: u64, more: bool) !Pick {
        const h = d.head;
        const ids = h.w.ids;
        const f = h.e.forward();
        const s = f.s;
        const n = ids.len;
        const sampled = if (rule) |r| r.temperature > 0 else false;
        if (!sampled) {
            try f.ops.argmax(h.logits, n, n, h.pick, 1);
            if (more) try d.rowStats();
            var out: [3]u32 = .{ 0, 0, @bitCast(@as(f32, 1.0)) };
            try f.ops.download(std.mem.sliceAsBytes(out[0 .. if (more) 3 else 1]), h.pick);
            try f.ops.s.synchronize();
            return .{ .token = ids[out[0]], .prob = if (more) 1.0 / @as(f32, @bitCast(out[2])) else 0 };
        }
        const r = rule.?;
        const want = if (r.top_k != 0) @min(n, @as(usize, r.top_k) + 8) else 0;
        var stats: [2]f32 = .{ 0, 1 };
        if (want > 0 and want <= q.state.max_candidates) {
            const t = f.ops.torch();
            try t.toF32(h.logits, s.logits32, n);
            try t.topk(s.logits32, n, 1, want, s.cand_values, s.cand_ids, s.topk_scratch);
            if (more) try d.rowStats();
            try f.ops.download(std.mem.sliceAsBytes(d.cand_values[0..want]), s.cand_values);
            try f.ops.download(std.mem.sliceAsBytes(d.cand_ids[0..want]), s.cand_ids);
            if (more) try f.ops.download(std.mem.sliceAsBytes(&stats), h.pick + 4);
            try f.ops.s.synchronize();
            var tokens: [q.state.max_candidates]u64 = undefined;
            for (d.values[0..want], d.cand_values[0..want], tokens[0..want], d.cand_ids[0..want]) |*v, x, *tok, col| {
                v.* = x;
                tok.* = ids[@intCast(col)];
            }
            const token = try lanes.sampling.choose(d.gpa, d.values[0..want], tokens[0..want], position, r);
            const at = std.mem.indexOfScalar(u64, tokens[0..want], token).?;
            return .{ .token = @intCast(token), .prob = probOf(d.cand_values[at], stats) };
        }
        if (more) try d.rowStats();
        try f.ops.download(std.mem.sliceAsBytes(d.row), h.logits);
        if (more) try f.ops.download(std.mem.sliceAsBytes(&stats), h.pick + 4);
        try f.ops.s.synchronize();
        const values = try d.gpa.alloc(f64, n);
        defer d.gpa.free(values);
        const tokens = try d.gpa.alloc(u64, n);
        defer d.gpa.free(tokens);
        for (values, tokens, d.row, ids) |*v, *tok, w, id| {
            v.* = @as(f32, @bitCast(@as(u32, w) << 16));
            tok.* = id;
        }
        const token = try lanes.sampling.choose(d.gpa, values, tokens, position, r);
        const at = std.mem.indexOfScalar(u64, tokens, token).?;
        return .{ .token = @intCast(token), .prob = probOf(@floatCast(values[at]), stats) };
    }

    /// Each stream's chain: the first draft from the absorbed row, the rest from the head's own outputs.
    fn hold(p: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = self(p);
        for (items) |it| {
            const l = d.lanes.get(it.stream) orelse return error.NotDrafting;
            const n = @min(it.depth, depth);
            try d.head.scoresOf(l.cache.last);
            var pick = try d.draw(it.stream.sampling, it.position, n > 1);
            l.tokens[0] = pick.token;
            var j: usize = 1;
            // the chain goes on while the last draft was confident (decode.mtp_round)
            while (j < n and @as(f64, pick.prob) >= confidence) : (j += 1) {
                try d.call(l, l.cache.last, l.tokens[j - 1 ..][0..1], l.cache.pos + j - 1);
                try d.head.scoresOf(l.cache.last);
                pick = try d.draw(it.stream.sampling, it.position + j, j + 1 < n);
                l.tokens[j] = pick.token;
            }
            l.count = j;
        }
    }

    fn heldCount(p: *anyopaque, s: *lanes.Stream) ?u32 {
        const l = self(p).lanes.get(s) orelse return null;
        return @intCast(l.count);
    }

    fn held(p: *anyopaque, streams: []const *lanes.Stream, out: []const []u32) anyerror!void {
        const d = self(p);
        for (streams, out) |s, o| {
            const l = d.lanes.get(s) orelse return error.NotDrafting;
            if (o.len > l.count) return error.ShortChain;
            @memcpy(o, l.tokens[0..o.len]);
        }
    }

    fn release(p: *anyopaque, s: *lanes.Stream) void {
        const d = self(p);
        if (d.lanes.fetchRemove(s)) |kv| {
            kv.value.cache.deinit();
            d.gpa.destroy(kv.value);
        }
    }
};
