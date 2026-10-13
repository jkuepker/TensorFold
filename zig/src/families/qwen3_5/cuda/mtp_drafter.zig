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

    /// Takes the head; times one chain step for the depth rule.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, head: *mtp.Head) !Drafter {
        var d: Drafter = .{ .gpa = gpa, .head = head };
        errdefer d.deinit();
        d.cand_values = try gpa.alloc(f32, q.state.max_candidates);
        d.cand_ids = try gpa.alloc(i64, q.state.max_candidates);
        d.values = try gpa.alloc(f64, q.state.max_candidates);
        d.row = try gpa.alloc(u16, head.w.ids.len);
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
    }

    pub fn drafter(d: *Drafter) dr.Drafter {
        return .{ .ptr = d, .vtable = &.{ .taps = taps, .facts = facts, .open = open, .absorb = absorb, .hold = hold, .held = held, .release = release } };
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
            _ = try d.draw(null, 1);
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
        return .{ .depth = depth, .step_ms = d.step_ms, .stream_bytes = mtp.Cache.bytes(d.head.e.w.g, d.head.e.context) };
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

    /// The target's draw rule on the head's logits, mapped to a token (decode.draft): the greedy column, or a keyed draw.
    fn draw(d: *Drafter, rule: ?lanes.Sampling, position: u64) !u32 {
        const h = d.head;
        const ids = h.w.ids;
        const f = h.e.forward();
        const s = f.s;
        const n = ids.len;
        const sampled = if (rule) |r| r.temperature > 0 else false;
        if (!sampled) {
            try f.ops.argmax(h.logits, n, n, h.pick, 1);
            var col: [1]u32 = undefined;
            try f.ops.download(std.mem.sliceAsBytes(&col), h.pick);
            try f.ops.s.synchronize();
            return ids[col[0]];
        }
        const r = rule.?;
        const want = if (r.top_k != 0) @min(n, @as(usize, r.top_k) + 8) else 0;
        if (want > 0 and want <= q.state.max_candidates) {
            const t = f.ops.torch();
            try t.toF32(h.logits, s.logits32, n);
            try t.topk(s.logits32, n, 1, want, s.cand_values, s.cand_ids, s.topk_scratch);
            try f.ops.download(std.mem.sliceAsBytes(d.cand_values[0..want]), s.cand_values);
            try f.ops.download(std.mem.sliceAsBytes(d.cand_ids[0..want]), s.cand_ids);
            try f.ops.s.synchronize();
            var tokens: [q.state.max_candidates]u64 = undefined;
            for (d.values[0..want], d.cand_values[0..want], tokens[0..want], d.cand_ids[0..want]) |*v, x, *tok, col| {
                v.* = x;
                tok.* = ids[@intCast(col)];
            }
            return @intCast(try lanes.sampling.choose(d.gpa, d.values[0..want], tokens[0..want], position, r));
        }
        try f.ops.download(std.mem.sliceAsBytes(d.row), h.logits);
        try f.ops.s.synchronize();
        const values = try d.gpa.alloc(f64, n);
        defer d.gpa.free(values);
        const tokens = try d.gpa.alloc(u64, n);
        defer d.gpa.free(tokens);
        for (values, tokens, d.row, ids) |*v, *tok, w, id| {
            v.* = @as(f32, @bitCast(@as(u32, w) << 16));
            tok.* = id;
        }
        return @intCast(try lanes.sampling.choose(d.gpa, values, tokens, position, r));
    }

    /// Each stream's chain: the first draft from the absorbed row, the rest from the head's own outputs.
    fn hold(p: *anyopaque, items: []const dr.Hold) anyerror!void {
        const d = self(p);
        for (items) |it| {
            const l = d.lanes.get(it.stream) orelse return error.NotDrafting;
            const n = @min(it.depth, depth);
            try d.head.scoresOf(l.cache.last);
            l.tokens[0] = try d.draw(it.stream.sampling, it.position);
            for (1..n) |j| {
                try d.call(l, l.cache.last, l.tokens[j - 1 ..][0..1], l.cache.pos + j - 1);
                try d.head.scoresOf(l.cache.last);
                l.tokens[j] = try d.draw(it.stream.sampling, it.position + j);
            }
            l.count = n;
        }
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
