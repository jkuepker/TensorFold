//! The Qwen3.6 MTP head's forward on CUDA (mtp.py Head._layer): [norm(embed(next token)) | norm(state)] through fc,
//! one attention layer over the head's own cache, routed experts, a norm, and the draft head over one row.

const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen27");
const weights = @import("mtp_weights.zig");
const moe = @import("moe.zig");

const c = q.config;
const st = q.state;
const tri = q.triton;
const rnd = q.round;
const bf = 2;
/// Rows the decode form of the experts takes a call (its fp32 pair outputs are sized for this many).
const moe_rows = 16;

/// The head's attention cache: slot t holds the key and value of position t; slots below `pos` are absorbed.
pub const Cache = struct {
    arena: st.Arena,
    k: u64,
    v: u64,
    last: u64, // the last absorbed row's normed output (hidden bf16): the state the next draft chain starts from
    capacity: usize,
    pos: usize = 0,

    pub fn bytes(g: c.Geometry, capacity: usize) usize {
        return 2 * std.mem.alignForward(usize, capacity * g.kvInner() * bf, 256) + std.mem.alignForward(usize, g.hidden * bf, 256);
    }

    pub fn init(d: *const cuda.Driver, g: c.Geometry, capacity: usize) !Cache {
        var a = try st.Arena.init(d, bytes(g, capacity));
        errdefer a.deinit();
        const n = capacity * g.kvInner() * bf;
        return .{ .arena = a, .k = try a.take(n), .v = try a.take(n), .last = try a.take(g.hidden * bf), .capacity = capacity };
    }

    pub fn deinit(x: *Cache) void {
        x.arena.deinit();
    }
};

/// A look at the call's norms in order (embedding, state, input, post, final): index, residual rows, normed rows, sums.
pub const Trace = struct {
    ctx: *anyopaque,
    norm: *const fn (ctx: *anyopaque, index: usize, h: u64, y: u64, xs: u64, rows: usize) anyerror!void,
};

pub const Head = struct {
    gpa: std.mem.Allocator,
    e: *q.Engine,
    w: weights.Head,
    block: *moe.Block,
    arena: st.Arena,
    xs2: u64, // the state half's group sums
    normed: u64, // the last call's normed rows (rows, hidden) bf16 and their group sums
    normed_xs: u64,
    logits: u64, // the draft head over one row, bf16 (the draft ids)
    dxs: u64, // that row's group sums
    pick: u64, // the argmax column, int32
    trace: ?Trace = null, // a test's look at each norm of a call (the oracle's numbering)
    pseudo: st.Seq, // the round layout's view of the head's cache: only the attention layers' pointers are read

    /// `w` is taken over (freed by deinit); `block` is the trunk's MoE block, one layer running at a time.
    pub fn init(gpa: std.mem.Allocator, e: *q.Engine, w: weights.Head, block: *moe.Block) !Head {
        const g = e.w.g;
        var h: Head = undefined;
        h.gpa = gpa;
        h.e = e;
        h.w = w;
        h.block = block;
        h.trace = null;
        const rows = e.scratch.rows;
        var plan: st.Arena = .{ .buf = null };
        try h.layout(&plan, g, rows);
        h.arena = try st.Arena.init(e.ctx.d, plan.used);
        errdefer h.arena.deinit();
        try h.layout(&h.arena, g, rows);
        h.pseudo = .{ .arena = undefined, .rec = &.{}, .conv = &.{}, .keys = &.{}, .values = &.{}, .head_in = 0, .head_xs = 0, .capacity = 0, .gpa = gpa };
        h.pseudo.keys = try gpa.alloc(u64, g.layers);
        errdefer gpa.free(h.pseudo.keys);
        h.pseudo.values = try gpa.alloc(u64, g.layers);
        @memset(h.pseudo.keys, 0);
        @memset(h.pseudo.values, 0);
        return h;
    }

    fn layout(h: *Head, a: *st.Arena, g: c.Geometry, rows: usize) !void {
        h.xs2 = try a.take(rows * g.hidden / 64 * 4);
        h.normed = try a.take(rows * g.hidden * bf);
        h.normed_xs = try a.take(rows * g.hidden / 64 * 4);
        h.logits = try a.take(h.w.draft.npad * bf);
        h.dxs = try a.take(g.hidden / 64 * 4);
        h.pick = try a.take(16);
    }

    pub fn deinit(h: *Head) void {
        h.gpa.free(h.pseudo.keys);
        h.gpa.free(h.pseudo.values);
        h.arena.deinit();
        h.w.deinit();
        h.* = undefined;
    }

    /// Device bytes the head holds beside the trunk (weights and buffers).
    pub fn bytes(h: *const Head) usize {
        return h.w.bytes() + (h.arena.buf orelse return 0).len;
    }

    fn attnShape(g: c.Geometry) tri.Attn {
        return .{ .heads = g.query_heads, .kv_heads = g.kv_heads, .dim = c.head_dim, .half = c.rotary_dim / 2 };
    }

    /// Rows at positions [p0, p0 + n): each a (state at t, token t + 1) pair attending to the cache's slots below p0 and
    /// to the rows before it. `states` (device, bf16 [n, hidden]) are the target's final normed rows. The normed outputs
    /// land in `normed` (with sums in `normed_xs`), the rows' keys and values at slots [p0, p0 + n) of `cache`.
    pub fn forward(h: *Head, cache: *Cache, states: u64, tokens: []const u32, p0: usize) !void {
        const n = tokens.len;
        const e = h.e;
        const g = e.w.g;
        const hid = g.hidden;
        const w = &h.w;
        var f = e.forward();
        const s = f.s;
        const t = f.t;
        const kv = g.kvInner() * bf;
        if (n == 0 or n > s.rows or p0 + n > cache.capacity) return error.MtpRowsPastCache;
        const wide = n > tri.max_nodes; // a prompt chunk: the prefill kernel, the rows' keys written first
        var round: rnd.Round = undefined;
        var ids = s.ids;
        var pos = s.pos;
        if (wide) {
            const host = try h.gpa.alloc(i32, 2 * n);
            defer h.gpa.free(host);
            for (tokens, 0..) |tok, r| {
                host[r] = @intCast(tok);
                host[n + r] = @intCast(p0 + r);
            }
            try f.ops.upload(s.ids, std.mem.sliceAsBytes(host[0..n]));
            try f.ops.upload(s.pos, std.mem.sliceAsBytes(host[n..]));
            try f.ops.s.synchronize(); // the host copy must outlive the async uploads
        } else {
            for (h.pseudo.keys, h.pseudo.values, 0..) |*k, *v, i| if (!c.linear(i)) {
                k.* = cache.k;
                v.* = cache.v;
            };
            h.pseudo.pos = p0;
            h.pseudo.capacity = cache.capacity;
            round = try rnd.stage(h.gpa, f.ops, t, s, g, &.{.{ .seq = &h.pseudo, .tokens = tokens }});
            ids = round.ids;
            pos = round.pos;
        }
        try t.embed(ids, e.w.embed.w, e.w.embed.s, e.w.embed.b, s.x, n, hid);
        try t.addRmsnorm(s.x, null, w.norm_e, s.x, s.y, s.xs, n, hid, c.eps);
        try h.look(0, s.x, s.y, s.xs, n);
        try t.addRmsnorm(states, null, w.norm_h, states, s.h, h.xs2, n, hid, c.eps);
        try h.look(1, states, s.h, h.xs2, n);
        try f.ops.dense(s.y, s.xs, w.fc_e, s.r, n);
        try f.ops.dense(s.h, h.xs2, w.fc_h, s.pending, n);
        // the two halves added in fp32 and rounded to bf16, then the input norm
        try t.addRmsnorm(s.r, s.pending, w.input_norm, s.x, s.y, s.xs, n, hid, c.eps);
        try h.look(2, s.x, s.y, s.xs, n);
        const a = w.attn;
        try f.ops.dense(s.y, s.xs, a.q, s.qg, n);
        try f.ops.dense(s.y, s.xs, a.k, s.key, n);
        try f.ops.dense(s.y, s.xs, a.v, s.value, n);
        try t.attnPrep(s.qg, s.key, a.q_norm, a.k_norm, pos, e.w.inv_freq, s.q_rot, s.prompt_keys, n, attnShape(g), c.eps);
        if (wide) {
            try f.ops.copy(cache.k + p0 * kv, s.prompt_keys, n * kv);
            try f.ops.copy(cache.v + p0 * kv, s.value, n * kv);
            const scale: f32 = @floatCast(std.math.pow(f64, c.head_dim, -0.5));
            try f.ops.prefillAttention(s.q_rot, cache.k, cache.v, s.attn, p0, n, g.query_heads, g.kv_heads, scale);
        } else {
            try t.attention(s.q_rot, s.prompt_keys, s.value, s.origin, round.offsets(g.layers - 1, 0), round.attn, .{ .o = s.part_o, .m = s.part_m, .l = s.part_l }, s.attn, attnShape(g));
            try f.ops.copy(cache.k + p0 * kv, s.prompt_keys, n * kv);
            try f.ops.copy(cache.v + p0 * kv, s.value, n * kv);
        }
        try t.gateMul(s.attn, s.qg, s.gated, s.gated_xs, n, g.query_heads, c.head_dim);
        try f.ops.dense(s.gated, s.gated_xs, a.o, s.r, n);
        try t.addRmsnorm(s.x, s.r, w.post_norm, s.h, s.y, s.xs, n, hid, c.eps);
        try h.look(3, s.h, s.y, s.xs, n);
        // the decode form of the experts at any row count, a few rows a call
        var at: usize = 0;
        while (at < n) : (at += moe_rows) {
            const take = @min(moe_rows, n - at);
            try h.block.run(w.moe, s.y + at * hid * bf, take, false, s.pending + at * hid * bf);
        }
        try t.addRmsnorm(s.h, s.pending, w.norm, s.x, h.normed, h.normed_xs, n, hid, c.eps);
        try h.look(4, s.x, h.normed, h.normed_xs, n);
    }

    fn look(h: *Head, index: usize, res: u64, y: u64, xs: u64, rows: usize) !void {
        if (h.trace) |t| try t.norm(t.ctx, index, res, y, xs, rows);
    }

    /// The draft head over row `row` of the last call's normed rows: logits (the draft ids) bf16 in `logits`.
    pub fn scores(h: *Head, row: usize) !void {
        const f = h.e.forward();
        const hid = h.e.w.g.hidden;
        try f.t.groupSums(h.normed + row * hid * bf, h.dxs, hid, 1, hid);
        try f.ops.dense(h.normed + row * hid * bf, h.dxs, h.w.draft, h.logits, 1);
    }

    /// The draft head over a stored normed row `at` (hidden bf16).
    pub fn scoresOf(h: *Head, at: u64) !void {
        const f = h.e.forward();
        const hid = h.e.w.g.hidden;
        try f.t.groupSums(at, h.dxs, hid, 1, hid);
        try f.ops.dense(at, h.dxs, h.w.draft, h.logits, 1);
    }
};
