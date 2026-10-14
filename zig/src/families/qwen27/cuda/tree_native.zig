//! Qwen3.6 tree attention from qwen_attention.cu (_shared, _tail, _merge), native where `exact` says it is byte-equal.

const std = @import("std");
const cuda = @import("cuda");
const tri = @import("triton.zig");

/// Which native kernels reproduce the captured Triton launches bit for bit (tf-cuda-test qwen-attention decides).
pub const exact: Exact = .{ .shared = true, .tail = true, .merge = true };

pub const Exact = struct { shared: bool, tail: bool, merge: bool };

/// The compute capability the byte-equality was shown on (GB10); the tensor-core dot's bits are an architecture's own.
pub const verified_capability: u32 = 121;

/// The shapes qwen_attention.cu is written for: 16 query heads over 2 key heads of 256 columns.
pub const heads = 16;
pub const kv_heads = 2;
pub const head_dim = 256;

/// qwen_attention.cu's Smem: two 64-key tiles of 264-wide rows, the 16 x 72 probabilities, and the row max and sum.
const smem_bytes: u32 = 2 * 64 * 264 * 2 + 16 * 72 * 2 + 2 * 4 * 16 * 4;

fn u(x: usize) u32 {
    return @intCast(x);
}

pub fn fits(a: tri.Attn) bool {
    return a.heads == heads and a.kv_heads == kv_heads and a.dim == head_dim;
}

pub const Native = struct {
    module: cuda.Module,
    shared_fn: cuda.Function,
    tail_fn: cuda.Function,
    merge_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver, capability: u32) !Native {
        if (capability != verified_capability) return error.UnverifiedArchitecture;
        var m = try cuda.Module.load(d, cuda.kernels.qwen_attention);
        errdefer m.unload();
        const shared_fn = try m.function("tf_qwen_attn_shared");
        const tail_fn = try m.function("tf_qwen_attn_tail");
        try shared_fn.allowDynamicShared(smem_bytes);
        try tail_fn.allowDynamicShared(smem_bytes);
        return .{ .module = m, .shared_fn = shared_fn, .tail_fn = tail_fn, .merge_fn = try m.function("tf_qwen_attn_merge") };
    }

    pub fn deinit(n: *Native) void {
        n.module.unload();
    }

    fn go(f: cuda.Function, s: cuda.Stream, grid: [3]u32, smem: u32, a: *cuda.Args) !void {
        // every kernel waits on the grid before it
        try cuda.launch.launch(f, .{ .grid = .{ .x = grid[0], .y = grid[1], .z = grid[2] }, .block = .{ .x = 128 }, .shared = smem, .pdl = true }, s, a);
    }

    /// attention._shared: the committed chunks and folded groups of every item, in 16-row query tiles.
    pub fn shared(n: Native, s: cuda.Stream, q: u64, base: u64, offs: u64, plan: tri.Tri.Plan, part: tri.Tri.Partials) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ q, base, base, offs, plan.streams, plan.items, part.o, part.m, part.l }) |v| a.add(v);
        a.add(@as(c_int, @intCast(plan.width)));
        try go(n.shared_fn, s, .{ u(plan.n_items * kv_heads), 1, 1 }, smem_bytes, &a);
    }

    /// attention._tail: each row's last committed keys and its own path, per key head and tail chunk.
    pub fn tail(n: Native, s: cuda.Stream, q: u64, kn: u64, vn: u64, base: u64, offs: u64, plan: tri.Tri.Plan, part: tri.Tri.Partials) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ q, kn, vn, base, base, offs, plan.streams, plan.rows, plan.paths, plan.depths, part.o, part.m, part.l }) |v| a.add(v);
        a.add(@as(c_int, @intCast(plan.width)));
        try go(n.tail_fn, s, .{ u(plan.width), kv_heads, 1 + (tri.max_nodes + tri.chunk - 1) / tri.chunk }, smem_bytes, &a);
    }

    /// attention._merge: every slot folded in order, o / l as bf16, 64 output columns a block.
    pub fn merge(n: Native, s: cuda.Stream, plan: tri.Tri.Plan, part: tri.Tri.Partials, out: u64) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ part.o, part.m, part.l, out, plan.streams, plan.rows }) |v| a.add(v);
        a.add(@as(c_int, @intCast(plan.width)));
        try go(n.merge_fn, s, .{ u(plan.width), kv_heads, head_dim / 64 }, 0, &a);
    }
};
