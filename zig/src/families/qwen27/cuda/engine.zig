//! The 27B's CUDA engine for one stream: the weights, kernels and scratch, prefill and greedy serial decode.

const std = @import("std");
const cuda = @import("cuda");
const c = @import("shape.zig");
const kern = @import("kernels.zig");
const tri = @import("triton.zig");
const tree_native = @import("tree_native.zig");
const st = @import("state.zig");
const wts = @import("weights.zig");
const Forward = @import("forward.zig").Forward;
const Part = @import("forward.zig").Part;

/// prefill.CHUNK: a prompt chunk's rows (prompt_rows on a GPU of 80 GB or more).
pub const prompt_rows = 4096;

/// prefill.chunks: even chunks of at most `size` rows of [start, end); any bounds give the same bits.
pub fn chunkBounds(start: usize, end: usize, size: usize, j: usize) [2]usize {
    const n = (end - start + size - 1) / size;
    return .{ start + (end - start) * j / n, start + (end - start) * (j + 1) / n };
}

pub const Options = struct { context: usize };

/// Where a prompt pass stops to let the cache keep its state: each of `at` (ascending) ends a chunk and `call` runs there.
pub const Keep = struct { at: []const u32, ctx: *anyopaque, call: *const fn (ctx: *anyopaque, at: u32) void };

/// A family's own weights and layer pieces beside this engine's: freed in place of `w` before the kernels they read.
pub const Owner = struct { ptr: *anyopaque, deinit: *const fn (*anyopaque) void };

pub const Engine = struct {
    gpa: std.mem.Allocator,
    ctx: *const cuda.Context,
    s: cuda.Stream,
    k: kern.Kernels,
    tree: ?tree_native.Native, // qwen_attention.cu where the GPU is the one it was shown byte-equal on, else the captured launches
    w: wts.Weights,
    scratch: st.Scratch,
    context: usize,
    hooks: ?Forward.Hooks = null, // a family's MLP in place of the dense one (the Qwen3.6 MoE)
    owner: ?Owner = null, // set when a family loaded `w` and owns it

    pub fn treeKernels(ctx: *const cuda.Context) !?tree_native.Native {
        if (!cuda.kernels.available) return null;
        return tree_native.Native.load(ctx.d, try ctx.capability()) catch |err| switch (err) {
            error.UnverifiedArchitecture => null,
            else => err,
        };
    }

    pub fn init(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, o: Options) !*Engine {
        const cfg = try c.Config.read(gpa, io, dir);
        const g = cfg.g;
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.ctx = ctx;
        e.context = o.context;
        e.s = try cuda.Stream.init(ctx.d, true);
        errdefer e.s.deinit();
        e.k = try kern.Kernels.load(gpa, io, ctx, kernels, g.linear_v_heads);
        errdefer e.k.deinit();
        e.tree = try treeKernels(ctx);
        errdefer if (e.tree) |*n| n.deinit();
        e.w = try wts.load(gpa, io, e.ops(), dir, cfg);
        errdefer e.w.deinit();
        e.scratch = try st.Scratch.init(gpa, ctx.d, g, prompt_rows, o.context);
        errdefer e.scratch.deinit();
        try e.s.synchronize();
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.scratch.deinit();
        if (e.owner) |o| o.deinit(o.ptr) else e.w.deinit();
        if (e.tree) |*n| n.deinit();
        e.k.deinit();
        e.s.deinit();
        e.gpa.destroy(e);
    }

    pub fn ops(e: *Engine) kern.Ops {
        return .{ .k = &e.k, .s = e.s };
    }

    pub fn forward(e: *Engine) Forward {
        var t = e.k.tri(e.s);
        t.tree = e.tree;
        return .{ .gpa = e.gpa, .w = &e.w, .ops = e.ops(), .t = t, .s = &e.scratch, .hooks = e.hooks };
    }

    /// A stream's state for `capacity` positions (its caches, recurrences and conv windows), empty.
    pub fn sequence(e: *Engine, capacity: usize) !st.Seq {
        return st.Seq.init(e.gpa, e.ops(), e.w.g, capacity);
    }

    /// prefill_state from an empty state: the prompt in even chunks of at most `prompt_rows`, its last row normed.
    pub fn prefill(e: *Engine, seq: *st.Seq, prompt: []const u32) !void {
        return e.prefillFrom(seq, prompt, 0, null, null);
    }

    /// `prefill`, every row's final normed state also written to `rows` (prompt.len x hidden bf16) for a draft head.
    pub fn prefillRows(e: *Engine, seq: *st.Seq, prompt: []const u32, rows: ?u64) !void {
        return e.prefillFrom(seq, prompt, 0, null, rows);
    }

    /// The prompt's rows from `from` (a restored state's position, 0: an empty one); each of `keep.at` ends a chunk and is called there; `rows` as in prefillRows.
    pub fn prefillFrom(e: *Engine, seq: *st.Seq, prompt: []const u32, from: usize, keep: ?Keep, rows: ?u64) !void {
        if (prompt.len == 0) return error.EmptyPrompt;
        if (prompt.len >= seq.capacity) return error.PromptTooLong;
        if (from == 0) try seq.reset(e.ops(), e.w.g) else if (seq.pos != from or from >= prompt.len) return error.NotAtMark;
        var f = e.forward();
        var start = from;
        var next: usize = 0; // the first mark past `start`
        while (start < prompt.len) {
            while (keep != null and next < keep.?.at.len and keep.?.at[next] <= start) next += 1;
            const marked = keep != null and next < keep.?.at.len and keep.?.at[next] <= prompt.len;
            const end: usize = if (marked) keep.?.at[next] else prompt.len;
            const n = (end - start + prompt_rows - 1) / prompt_rows;
            for (0..n) |j| {
                const b = chunkBounds(start, end, prompt_rows, j);
                f.every = if (rows) |r| r + b[0] * e.w.g.hidden * 2 else null;
                try f.chunk(prompt[b[0]..b[1]], seq, b[1] == prompt.len);
            }
            if (marked) keep.?.call(keep.?.ctx, @intCast(end));
            start = end;
        }
    }

    /// decode.serial_decode, greedy: one-row windows from `pending` until `count` tokens or an end token.
    pub fn decode(e: *Engine, seq: *st.Seq, pending: u32, count: usize, out: *std.ArrayList(u32)) !void {
        var f = e.forward();
        try out.append(e.gpa, pending);
        var pick: [1]u32 = undefined;
        while (out.items.len < count and !e.w.isEos(out.items[out.items.len - 1])) {
            const last = out.items[out.items.len - 1];
            const part = [_]Part{.{ .seq = seq, .tokens = &.{last} }};
            try f.round(&part);
            try f.picks(1, &pick);
            try f.commit(&part, &.{&.{0}});
            try out.append(e.gpa, pick[0]);
        }
    }

    /// The prompt, then up to `max_tokens` greedy tokens (the first included), as generate(draft=False) gives them.
    pub fn generate(e: *Engine, seq: *st.Seq, prompt: []const u32, max_tokens: usize, out: *std.ArrayList(u32)) !void {
        const room = @min(max_tokens, seq.capacity - prompt.len);
        try e.prefill(seq, prompt);
        var f = e.forward();
        try f.head(seq);
        var pick: [1]u32 = undefined;
        try f.picks(1, &pick);
        try e.decode(seq, pick[0], @max(1, room), out);
    }
};

test "prefill.chunks splits evenly" {
    try std.testing.expectEqual([2]usize{ 0, 2539 }, chunkBounds(0, 5078, 4096, 0));
    try std.testing.expectEqual([2]usize{ 2539, 5078 }, chunkBounds(0, 5078, 4096, 1));
    try std.testing.expectEqual([2]usize{ 0, 40 }, chunkBounds(0, 40, 4096, 0));
}
