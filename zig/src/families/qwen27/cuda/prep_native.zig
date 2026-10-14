//! Qwen3.6's per-layer prep kernels (glue.gdn_pre, glue.attn_prep) from qwen_prep.cu, native where `exact` says so.

const std = @import("std");
const cuda = @import("cuda");

/// native kernels reproduce Triton launches bit for bit (tf-cuda-test qwen-prep decides); rest replay.
pub const exact: Exact = .{ .gdn_pre = true, .attn_prep = true };
pub const Exact = struct { gdn_pre: bool, attn_prep: bool };

/// The compute capability the byte-equality was shown on (GB10); libdevice-class math is architecture-specific.
pub const verified_capability: u32 = 121;

/// The shapes the captured launches and this test cover (Qwen3.6 35B-A3B, MiniCPM-V 4.7); any other stays on replay.
pub fn gdnFits(kh: usize, vh: usize, dk: usize, c: usize, keep: usize) bool {
    return c == 8192 and kh == 16 and vh == 32 and dk == 128 and keep == 3;
}

pub fn attnFits(heads: usize, kv_heads: usize, dim: usize, half: usize) bool {
    return heads == 16 and kv_heads == 2 and dim == 256 and half == 32;
}

fn int(v: usize) c_int {
    return @intCast(v);
}

/// qwen_prep.cu's kernels.
pub const Native = struct {
    module: cuda.Module,
    gdn_fn: cuda.Function,
    attn_fn: cuda.Function,

    pub fn load(d: *const cuda.Driver, capability: u32) !Native {
        return loadImage(d, capability, cuda.kernels.qwen_prep);
    }

    /// `load` from another fatbin of the same kernels (the test's sensitivity controls).
    pub fn loadImage(d: *const cuda.Driver, capability: u32, image: []const u8) !Native {
        if (capability != verified_capability) return error.UnverifiedArchitecture;
        var m = try cuda.Module.load(d, image);
        errdefer m.unload();
        return .{ .module = m, .gdn_fn = try m.function("tf_qwen_gdn_pre"), .attn_fn = try m.function("tf_qwen_attn_prep") };
    }

    pub fn deinit(n: *Native) void {
        n.module.unload();
    }

    fn go(f: cuda.Function, s: cuda.Stream, grid: [2]u32, a: *cuda.Args) !void {
        try cuda.launch.launch(f, .{ .grid = .{ .x = grid[0], .y = grid[1], .z = 1 }, .block = .{ .x = 64 }, .pdl = true }, s, a); // both kernels wait on the grid before them
    }

    /// glue.gdn_pre: conv over each row's window [state; qkv], q/k norms, g, beta; `sid`: per-row stream (MULTI).
    pub fn gdnPre(n: Native, s: cuda.Stream, qkv: u64, cs: u64, cw: u64, win: u64, sid: ?u64, a: u64, b: u64, alog: u64, dtb: u64, q: u64, k: u64, v: u64, g: u64, beta: u64, rows: usize, kh: usize, vh: usize, dk: usize, c: usize, keep: usize) !void {
        if (!gdnFits(kh, vh, dk, c, keep)) return error.PrepShape;
        var args: cuda.Args = .{};
        for ([_]u64{ qkv, cs, cw, win, a, b, alog, dtb, q, k, v, g, beta, sid orelse win }) |p| args.add(p);
        for ([_]usize{ c, kh, vh, keep, @intFromBool(sid != null) }) |x| args.add(int(x));
        try go(n.gdn_fn, s, .{ @intCast(rows), @intCast(2 * kh + vh) }, &args);
    }

    /// glue.attn_prep: q/k RMS norm then rotary, text positions (MROPE is not served); `half` rotary pairs.
    pub fn attnPrep(n: Native, s: cuda.Stream, qg: u64, kv: u64, qn: u64, kn: u64, pos: u64, inv: u64, qout: u64, kout: u64, rows: usize, heads: usize, kv_heads: usize, dim: usize, half: usize, eps: f32) !void {
        if (!attnFits(heads, kv_heads, dim, half)) return error.PrepShape;
        var args: cuda.Args = .{};
        for ([_]u64{ qg, kv, qn, kn, pos, inv, qout, kout }) |p| args.add(p);
        args.add(eps);
        for ([_]usize{ heads, kv_heads, dim, half }) |x| args.add(int(x));
        try go(n.attn_fn, s, .{ @intCast(rows), @intCast(heads + kv_heads) }, &args);
    }
};
