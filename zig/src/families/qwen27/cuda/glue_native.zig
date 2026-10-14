//! Qwen3.6's row glue as built-in kernels (qwen_glue.cu, embed_rows.cu), used where tf-cuda-test qwen-glue shows it.

const std = @import("std");
const cuda = @import("cuda");

/// Which native kernels reproduce the captured Triton launches bit for bit on every served shape; the rest replay.
pub const exact: Exact = .{ .embed = true, .add_rmsnorm = true, .gate_mul = true, .gated_norm = true, .group_sums = true, .paths = true };

pub const Exact = struct { embed: bool, add_rmsnorm: bool, gate_mul: bool, gated_norm: bool, group_sums: bool, paths: bool };

/// The compute capability the byte-equality was shown on (GB10): MUFU and div.full expansion are architecture-specific.
pub const verified_capability: u32 = 121;

/// The shapes the byte-equality was shown on (the captured constexprs): widths, heads, and the paths' depth cap.
pub const hidden = 2048;
pub const value_heads = 32;
pub const value_dim = 128;
pub const query_heads = 16;
pub const head_dim = 256;
pub const max_depth = 128;

fn u(x: usize) u32 {
    return @intCast(x);
}

fn cdiv(a: usize, b: usize) u32 {
    return u((a + b - 1) / b);
}

fn aligned(ptrs: []const u64) bool {
    for (ptrs) |p| if (p % 16 != 0) return false;
    return true;
}

pub const Native = struct {
    module: cuda.Module,
    emb: cuda.embed_rows.EmbedRows,
    rms_fn: cuda.Function,
    rms_res_fn: cuda.Function,
    gated_fn: cuda.Function,
    gate_fn: cuda.Function,
    sums_fn: cuda.Function,
    paths_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver, capability: u32) !Native {
        if (capability != verified_capability) return error.UnverifiedArchitecture;
        var m = try cuda.Module.load(d, cuda.kernels.qwen_glue);
        errdefer m.unload();
        var emb = try cuda.embed_rows.EmbedRows.load(d);
        errdefer emb.unload();
        return .{
            .module = m,
            .emb = emb,
            .rms_fn = try m.function("tf_qwen_add_rmsnorm"),
            .rms_res_fn = try m.function("tf_qwen_add_rmsnorm_res"),
            .gated_fn = try m.function("tf_qwen_gated_norm"),
            .gate_fn = try m.function("tf_qwen_gate_mul"),
            .sums_fn = try m.function("tf_qwen_group_sums"),
            .paths_fn = try m.function("tf_qwen_paths"),
        };
    }

    pub fn deinit(n: *Native) void {
        n.emb.unload();
        n.module.unload();
    }

    // every qwen_glue.cu kernel waits on the grid before it
    fn go(n: Native, s: cuda.Stream, f: cuda.Function, grid: [2]u32, block: u32, a: *cuda.Args) !void {
        _ = n;
        try cuda.launch.launch(f, .{ .grid = .{ .x = grid[0], .y = grid[1], .z = 1 }, .block = .{ .x = block }, .pdl = true }, s, a);
    }

    pub fn fitsEmbed(d: usize, out: u64) bool {
        return d == hidden and out % 16 == 0;
    }

    /// embed_rows.cu's 4-bit affine rows: bf16(q * scale + bias) of each id's table row.
    pub fn embed(n: *const Native, s: cuda.Stream, ids: u64, w: u64, scales: u64, biases: u64, out: u64, rows: usize, d: usize) !void {
        try n.emb.affine4(s, ids, w, scales, biases, out, rows, d);
    }

    pub fn fitsAddRmsnorm(ptrs: []const u64, d: usize) bool {
        return d == hidden and aligned(ptrs);
    }

    /// h = bf16(x + r), y = bf16(rmsnorm(h) * w), xs = y's 64-group sums; without `r` the norm reads x, h unwritten.
    pub fn addRmsnorm(n: Native, s: cuda.Stream, x: u64, r: ?u64, w: u64, h: u64, y: u64, xs: u64, rows: usize, eps: f32) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(x);
        if (r) |rp| a.add(rp);
        a.add(w);
        if (r != null) a.add(h);
        a.add(y);
        a.add(xs);
        a.add(eps);
        try n.go(s, if (r != null) n.rms_res_fn else n.rms_fn, .{ u(rows), 1 }, 256, &a);
    }

    pub fn fitsGatedNorm(ptrs: []const u64, vh: usize, dv: usize) bool {
        return vh == value_heads and dv == value_dim and aligned(ptrs);
    }

    /// out = bf16(silu(z) * rmsnorm(y) * w) per (row, value head), xs = its 64-group sums.
    pub fn gatedNorm(n: Native, s: cuda.Stream, y: u64, z: u64, w: u64, out: u64, xs: u64, rows: usize, vh: usize, eps: f32) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        for ([_]u64{ y, z, w, out, xs }) |v| a.add(v);
        a.add(eps);
        a.add(@as(c_int, @intCast(vh)));
        try n.go(s, n.gated_fn, .{ u(rows), u(vh) }, 32, &a);
    }

    pub fn fitsGateMul(ptrs: []const u64, heads: usize, dim: usize) bool {
        return heads == query_heads and dim == head_dim and aligned(ptrs);
    }

    /// out = bf16(o * sigmoid(gate)) from the [q | gate] rows, xs = its 64-group sums.
    pub fn gateMul(n: Native, s: cuda.Stream, o: u64, qg: u64, out: u64, xs: u64, rows: usize, heads: usize) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        for ([_]u64{ o, qg, out, xs }) |v| a.add(v);
        a.add(@as(c_int, @intCast(heads)));
        try n.go(s, n.gate_fn, .{ u(rows), u(heads) }, 64, &a);
    }

    pub fn fitsGroupSums(ptrs: []const u64, ld: usize, k: usize) bool {
        return k == hidden and ld % 16 == 0 and ld >= k and aligned(ptrs);
    }

    /// Each row's 64-group input sums in fp32, rows strided by `ld`.
    pub fn groupSums(n: Native, s: cuda.Stream, x: u64, xs: u64, ld: usize, rows: usize, k: usize) !void {
        if (rows == 0) return;
        var a: cuda.Args = .{};
        a.add(x);
        a.add(xs);
        a.add(@as(c_int, @intCast(ld)));
        a.add(@as(c_int, @intCast(k / 64)));
        try n.go(s, n.sums_fn, .{ u(rows), cdiv(k / 64, 16) }, 128, &a);
    }

    pub fn fitsPaths(rows: usize) bool {
        return rows > 0;
    }

    /// Each row's root-to-row window rows (PATHS, 128 slots a row) and depth from the parent table.
    pub fn paths(n: Native, s: cuda.Stream, parents: u64, out: u64, depths: u64, rows: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ parents, out, depths }) |v| a.add(v);
        a.add(@as(c_int, @intCast(rows)));
        try n.go(s, n.paths_fn, .{ cdiv(rows, 128), 1 }, 128, &a);
    }
};
