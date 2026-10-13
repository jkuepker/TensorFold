//! A MoE layer's router rows and expert stacks on the GPU in the Python engine's blocks (experts.make).

const std = @import("std");
const cuda = @import("cuda");
const qwen35 = @import("qwen35");
const w27 = @import("qwen27").weights;
const shape = @import("qwen27").config;
const pack = @import("pack.zig");

const Blocks = cuda.experts.Affine4Experts;
const block_bytes = 288 * 4; // one (32 outputs, 64 inputs) block of an expert matrix

/// One layer's routing and experts: slot `experts` of each stack is the shared expert.
pub const Moe = struct {
    /// [E + 1, hidden] bf16: the routers' rows, the shared expert's gate row last.
    router: u64,
    /// [E + 1][width / 32][hidden / 64][2][288] int32: a block's gate half, then its up half.
    up: u64,
    /// [E + 1][hidden / 32][width / 64][288] int32.
    down: u64,
    router_bytes: usize,
    up_bytes: usize,
    down_bytes: usize,
};

fn rows(comptime T: type, bytes: []const u8) []const T {
    return std.mem.bytesAsSlice(T, bytes);
}

/// `height` rows of `width` bytes, `pitch` apart in host memory, laid end to end at `dst`.
fn fromHost(L: *w27.Loader, dst: u64, src: []const u8, pitch: usize, width: usize, height: usize) !void {
    const m: cuda.abi.Memcpy2D = .{ .src_type = 1, .src_host = src.ptr, .src = 0, .src_pitch = pitch, .dst = dst, .dst_pitch = width, .width = width, .height = height };
    try L.pack.d.check(L.pack.d.api.cuMemcpy2DAsync_v2(&m, L.pack.s.handle), "cuMemcpy2DAsync");
}

/// `height` blocks end to end at `src`, written `pitch` apart from `dst`.
fn spread(L: *w27.Loader, dst: u64, pitch: usize, src: u64, height: usize) !void {
    const m: cuda.abi.Memcpy2D = .{ .src = src, .src_pitch = block_bytes, .dst = dst, .dst_pitch = pitch, .width = block_bytes, .height = height };
    try L.pack.d.check(L.pack.d.api.cuMemcpy2DAsync_v2(&m, L.pack.s.handle), "cuMemcpy2DAsync");
}

const Mlx = @FieldType(qwen35.host.Projection, "mlx");

fn mlx(p: qwen35.host.Projection, bits: u8, group: u16) !Mlx {
    const x = switch (p) {
        .mlx => |x| x,
        else => return error.UnsupportedQuantization,
    };
    if (x.bits != bits or x.group != group) return error.UnsupportedQuantization;
    if (x.scales.dtype != .bf16 or x.biases.dtype != .bf16) return error.UnsupportedQuantization;
    return x;
}

/// The checked stacks of one layer: every expert (shared last) in 4-bit group-64 words, bf16 scales and biases.
const Stacks = struct { fused: Mlx, down: Mlx };

fn stacks(r: qwen35.host.Routed, g: shape.Geometry, moe: shape.Moe) !Stacks {
    const e = r.experts;
    if (!e.gated or e.count != moe.experts + 1 or e.width != moe.width or e.dims != g.hidden) return error.UnexpectedTensor;
    if (r.top_k != moe.top_k or r.router.dtype != .bf16 or r.router.shape[0] != e.count or r.router.shape[1] != g.hidden) return error.UnexpectedTensor;
    const fused = try mlx(e.fused, 4, 64);
    const down = try mlx(e.down, 4, 64);
    const k8 = g.hidden / 8;
    if (fused.words.shape[0] != e.count or fused.words.shape[1] != 2 * e.width or fused.words.shape[2] != k8) return error.UnexpectedTensor;
    if (down.words.shape[0] != e.count or down.words.shape[1] != g.hidden or down.words.shape[2] != e.width / 8) return error.UnexpectedTensor;
    return .{ .fused = fused, .down = down };
}

/// A layer's router and experts uploaded, packed and checked; `r`'s host memory is read by the time this returns
/// but the copies may be queued: settle the loader before freeing it.
pub fn upload(L: *w27.Loader, k: *const pack.Kernels, r: qwen35.host.Routed, g: shape.Geometry, moe: shape.Moe) !Moe {
    const s = try stacks(r, g, moe);
    const e = r.experts.count;
    const w = r.experts.width;
    const d = g.hidden;
    var out: Moe = .{ .router = try L.alloc(r.router.bytes.len), .up = 0, .down = 0, .router_bytes = r.router.bytes.len, .up_bytes = 2 * Blocks.bytes(e, w, d), .down_bytes = Blocks.bytes(e, d, w) };
    try L.pack.upload(out.router, r.router.bytes);
    out.up = try L.alloc(out.up_bytes);
    out.down = try L.alloc(out.down_bytes);
    // scratch: gate's and up's rows as stored, then the two packed halves
    const raw = [3]usize{ e * w * (d / 8) * 4, e * w * (d / 64) * 2, e * w * (d / 64) * 2 };
    const raw_half = raw[0] + raw[1] + raw[2];
    const base = try L.tmp(2 * raw_half + 2 * Blocks.bytes(e, w, d));
    const packed_at = base + 2 * raw_half;
    const tensors = [3]qwen35.host.Tensor{ s.fused.words, s.fused.scales, s.fused.biases };
    for (0..2) |half| {
        var at = base + half * raw_half;
        for (tensors, raw) |t, size| {
            const row = t.bytes.len / (e * 2 * w);
            try fromHost(L, at, t.bytes[half * w * row ..], 2 * w * row, w * row, e);
            at += size;
        }
        const start = base + half * raw_half;
        try k.experts(start, start + raw[0], start + raw[0] + raw[1], packed_at + half * Blocks.bytes(e, w, d), e, w, d);
    }
    for (0..2) |half| try spread(L, out.up + half * block_bytes, 2 * block_bytes, packed_at + half * Blocks.bytes(e, w, d), e * (w / 32) * (d / 64));
    // down: n = hidden outputs from k = width inputs, packed straight into place
    const dtensors = [3]qwen35.host.Tensor{ s.down.words, s.down.scales, s.down.biases };
    const dsizes = [3]usize{ dtensors[0].bytes.len, dtensors[1].bytes.len, dtensors[2].bytes.len };
    const dbase = try L.tmp(dsizes[0] + dsizes[1] + dsizes[2]);
    var at = dbase;
    for (dtensors, dsizes) |t, size| {
        try L.pack.upload(at, t.bytes);
        at += size;
    }
    try k.experts(dbase, dbase + dsizes[0], dbase + dsizes[0] + dsizes[1], out.down, e, d, w);
    return out;
}
