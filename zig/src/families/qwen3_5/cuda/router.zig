//! Qwen3.6's MoE router on CUDA: fp32 expert logits, softmax top-k, slot-order combine (cubins or qwen_route.cu).

const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;

/// Which native kernels reproduce the Triton launches bit for bit (tf-cuda-test qwen35-router decides).
pub const exact: Exact = .{ .logits = true, .select = true, .combine = true };

pub const Exact = struct { logits: bool, select: bool, combine: bool };

/// The compute capability the byte-equality was shown on (GB10); the tensor-core dot's bits are an architecture's own.
pub const verified_capability: u32 = 121;

/// The layer shape: hidden width, routed experts (the shared expert's gate row follows them), picks a row.
pub const Shape = struct { d: usize, experts: usize, top_k: usize };

fn u(x: usize) u32 {
    return @intCast(x);
}

fn cdiv(a: usize, b: usize) u32 {
    return u((a + b - 1) / b);
}

fn pow2(n: usize) usize {
    return std.math.ceilPowerOfTwo(usize, n) catch unreachable;
}

fn ci(name: []const u8, v: usize) aot.Const {
    return aot.ci(name, @intCast(v));
}

/// moe._tile: rows a router program takes, 16 to 128.
pub fn tile(m: usize) usize {
    inline for (.{ 16, 32, 64, 128 }) |b| if (m <= b) return b;
    return 128;
}

/// The captured Triton launches, with moe.py's grids and constexprs; pointers must be 16-aligned.
pub const Replay = struct {
    set: *const aot.Set,
    s: cuda.Stream,

    /// moe.router: logits [m, experts + 1] fp32 of x [m, d] bf16 against the rows [experts + 1, d] bf16.
    pub fn logits(r: Replay, x: u64, w: u64, out: u64, m: usize, sh: Shape) !void {
        const ne = sh.experts + 1;
        const bm = tile(m);
        const be: usize = if (bm == 16) 32 else 64;
        const bk: usize = if (bm == 16) 256 else 64;
        try r.set.run(r.s, "_router", .{ cdiv(m, bm), cdiv(ne, be), 1 }, &.{
            aot.ptr("X", "*bf16", x), aot.ptr("W", "*bf16", w), aot.ptr("OUT", "*fp32", out), aot.int("M", @intCast(m)), aot.int("x_stride", @intCast(sh.d)),
        }, &.{ ci("D", sh.d), ci("NE", ne), ci("BM", bm), ci("BLOCK_E", be), ci("BK", bk) });
    }

    /// moe.select_rows: each row's picks (int32) and weights (fp32), the shared expert as the last slot.
    pub fn select(r: Replay, l: u64, pick: u64, wts: u64, rows: usize, sh: Shape) !void {
        try r.set.run(r.s, "_topk_rows", .{ u(rows), 1, 1 }, &.{
            aot.ptr("L", "*fp32", l), aot.ptr("PICK", "*i32", pick), aot.ptr("WTS", "*fp32", wts),
        }, &.{ ci("NE", sh.experts), ci("NL", sh.experts + 1), ci("TOPK", sh.top_k), ci("SLOTS", sh.top_k + 1), ci("BLOCK", pow2(sh.experts + 1)), ci("SLOTP", pow2(sh.top_k + 1)) });
    }

    /// moe.combine: y [rows, slots, d] (fp32, or bf16 in prefill) and wts [rows, slots] fp32 into out [rows, d] bf16.
    pub fn combine(r: Replay, y: u64, y_bf16: bool, wts: u64, out: u64, rows: usize, sh: Shape) !void {
        try r.set.run(r.s, "_combine", .{ u(rows), cdiv(sh.d, 512), 1 }, &.{
            aot.ptr("Y", if (y_bf16) "*bf16" else "*fp32", y), aot.ptr("W", "*fp32", wts), aot.ptr("OUT", "*bf16", out),
        }, &.{ ci("S", sh.top_k + 1), ci("D", sh.d), ci("BLOCK", 512) });
    }
};

/// K columns the native router stages a step, and the dynamic shared bytes they take (qwen_route.cu's KC, STAGES, PAD).
pub const stage_columns = 256;
const router_shared: u32 = 3 * (16 + 32) * (stage_columns + 8) * 2;

/// qwen_route.cu's kernels.
pub const Native = struct {
    module: cuda.Module,
    s: cuda.Stream,
    router_fn: cuda.Function,
    topk_fn: cuda.Function,
    combine_f32: cuda.Function,
    combine_bf16: cuda.Function,

    pub fn load(d: *const cuda.Driver, s: cuda.Stream, capability: u32) !Native {
        if (capability != verified_capability) return error.UnverifiedArchitecture;
        var m = try cuda.Module.load(d, cuda.kernels.qwen_route);
        errdefer m.unload();
        const router_fn = try m.function("tf_qwen_router");
        try router_fn.allowDynamicShared(router_shared);
        return .{
            .module = m,
            .s = s,
            .router_fn = router_fn,
            .topk_fn = try m.function("tf_qwen_topk"),
            .combine_f32 = try m.function("tf_qwen_combine_f32"),
            .combine_bf16 = try m.function("tf_qwen_combine_bf16"),
        };
    }

    pub fn deinit(n: *Native) void {
        n.module.unload();
    }

    fn go(n: Native, f: cuda.Function, grid: [2]u32, block: u32, shared: u32, a: *cuda.Args) !void {
        // every qwen_route.cu kernel waits on the grid before it
        try cuda.launch.launch(f, .{ .grid = .{ .x = grid[0], .y = grid[1], .z = 1 }, .block = .{ .x = block }, .shared = shared, .pdl = true }, n.s, a);
    }

    pub fn logits(n: Native, x: u64, w: u64, out: u64, m: usize, sh: Shape) !void {
        if (sh.d % stage_columns != 0) return error.RouterShape;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(w);
        a.add(out);
        for ([_]usize{ m, sh.d, sh.d, sh.experts + 1 }) |v| a.add(@as(c_int, @intCast(v)));
        try n.go(n.router_fn, .{ cdiv(m, 16), cdiv(sh.experts + 1, 32) }, 128, router_shared, &a);
    }

    pub fn select(n: Native, l: u64, pick: u64, wts: u64, rows: usize, sh: Shape) !void {
        if (sh.experts > 256 or sh.top_k > 8) return error.RouterShape;
        var a: cuda.Args = .{};
        a.add(l);
        a.add(pick);
        a.add(wts);
        for ([_]usize{ sh.experts, sh.experts + 1, sh.top_k }) |v| a.add(@as(c_int, @intCast(v)));
        try n.go(n.topk_fn, .{ u(rows), 1 }, 32, 0, &a);
    }

    pub fn combine(n: Native, y: u64, y_bf16: bool, wts: u64, out: u64, rows: usize, sh: Shape) !void {
        if (sh.d % 4 != 0) return error.RouterShape;
        var a: cuda.Args = .{};
        a.add(y);
        a.add(wts);
        a.add(out);
        for ([_]usize{ sh.top_k + 1, sh.d }) |v| a.add(@as(c_int, @intCast(v)));
        try n.go(if (y_bf16) n.combine_bf16 else n.combine_f32, .{ u(rows), cdiv(sh.d, 512) }, 128, 0, &a);
    }
};

/// Each step on its exact path: the native kernel where `exact` says it is byte-equal, else the captured Triton launch.
pub const Router = struct {
    replay: ?Replay,
    native: ?Native,

    pub fn logits(r: Router, x: u64, w: u64, out: u64, m: usize, sh: Shape) !void {
        if (r.native) |n| if (exact.logits) return n.logits(x, w, out, m, sh);
        return (r.replay orelse return error.NoExactRouter).logits(x, w, out, m, sh);
    }

    pub fn select(r: Router, l: u64, pick: u64, wts: u64, rows: usize, sh: Shape) !void {
        if (r.native) |n| if (exact.select) return n.select(l, pick, wts, rows, sh);
        return (r.replay orelse return error.NoExactRouter).select(l, pick, wts, rows, sh);
    }

    pub fn combine(r: Router, y: u64, y_bf16: bool, wts: u64, out: u64, rows: usize, sh: Shape) !void {
        if (r.native) |n| if (exact.combine) return n.combine(y, y_bf16, wts, out, rows, sh);
        return (r.replay orelse return error.NoExactRouter).combine(y, y_bf16, wts, out, rows, sh);
    }
};

test "the router tile follows moe._tile" {
    try std.testing.expectEqual(@as(usize, 16), tile(1));
    try std.testing.expectEqual(@as(usize, 16), tile(16));
    try std.testing.expectEqual(@as(usize, 32), tile(17));
    try std.testing.expectEqual(@as(usize, 128), tile(100));
    try std.testing.expectEqual(@as(usize, 128), tile(2048));
}
