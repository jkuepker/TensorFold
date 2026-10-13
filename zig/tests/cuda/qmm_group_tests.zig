//! qmm_group's fused launch against one launch a projection, byte for byte, on random 4-bit weights at the Qwen3.6 shapes.

const std = @import("std");
const cuda = @import("cuda");
const qwen27 = @import("qwen27");
const check = @import("check.zig");

const kern = qwen27.kernels;

/// A random tiled projection (any bytes are valid words; scales and biases are small bf16 values).
fn weight(gpu: check.Gpu, rng: std.Random, n: usize, k: usize, bufs: *std.ArrayList(cuda.DeviceBuffer)) !kern.QLinear {
    const sizes = cuda.qlinear.Affine4.layout(n, k);
    const w = try gpu.gpa.alloc(u8, sizes.words);
    defer gpu.gpa.free(w);
    rng.bytes(w);
    const sb = try gpu.gpa.alloc(u16, sizes.scales / 2);
    defer gpu.gpa.free(sb);
    var q: kern.QLinear = .{ .w = 0, .s = 0, .b = 0, .n = n, .k = k, .npad = sizes.npad };
    for ([_]*u64{ &q.w, &q.s, &q.b }, 0..) |dst, i| {
        const b = try cuda.DeviceBuffer.alloc(gpu.d, if (i == 0) w.len else sizes.scales);
        try bufs.append(gpu.gpa, b);
        if (i == 0) {
            try b.upload(0, w);
        } else {
            for (sb) |*v| v.* = @truncate(@as(u32, @bitCast(@as(f32, @floatFromInt(rng.intRangeLessThan(i32, -64, 64))) / 256.0)) >> 16);
            try b.upload(0, std.mem.sliceAsBytes(sb));
        }
        dst.* = b.ptr;
    }
    return q;
}

pub fn run(gpu: check.Gpu, kernels: []const u8) !void {
    const gpa = gpu.gpa;
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var k = try kern.Kernels.load(gpa, gpu.io, gpu.ctx, kernels, 32);
    defer k.deinit();
    const ops: kern.Ops = .{ .k = &k, .s = stream };
    var prng = std.Random.DefaultPrng.init(0x51ab);
    const rng = prng.random();
    var bufs: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer {
        for (bufs.items) |*b| b.free();
        bufs.deinit(gpa);
    }
    const hidden = 2048;
    // GDN qkv, z, b, a and attention q, k, v at Qwen3.6's shapes
    const gdn = [_][2]usize{ .{ 8192, hidden }, .{ 4096, hidden }, .{ 32, hidden }, .{ 32, hidden } };
    const attn = [_][2]usize{ .{ 8192, hidden }, .{ 512, hidden }, .{ 512, hidden } };
    const sets = [_][]const [2]usize{ &gdn, &attn, gdn[0..2], gdn[2..4], &.{ .{ 4096, 4096 }, .{ 2048, 4096 }, .{ 64, 4096 } } };
    const row_counts = [_]usize{ 1, 2, 3, 4, 5, 8, 16, 17, 32, 33, 64, 65, 80 };
    var weights: [4]kern.QLinear = undefined;
    var failed = false;
    for (sets, 0..) |set, si| {
        for (set, 0..) |nk, i| weights[i] = try weight(gpu, rng, nk[0], nk[1], &bufs);
        const kk = set[0][1];
        for (row_counts) |rows| {
            var x = try cuda.DeviceBuffer.alloc(gpu.d, rows * kk * 2);
            defer x.free();
            const xh = try gpa.alloc(u16, rows * kk);
            defer gpa.free(xh);
            for (xh) |*v| v.* = @truncate(@as(u32, @bitCast(@as(f32, @floatFromInt(rng.intRangeLessThan(i32, -128, 128))) / 128.0)) >> 16);
            try x.upload(0, std.mem.sliceAsBytes(xh));
            var xs = try cuda.DeviceBuffer.alloc(gpu.d, rows * (kk / 64) * 4);
            defer xs.free();
            const xsh = try gpa.alloc(f32, rows * (kk / 64));
            defer gpa.free(xsh);
            for (xsh) |*v| v.* = @as(f32, @floatFromInt(rng.intRangeLessThan(i32, -64, 64))) / 16.0;
            try xs.upload(0, std.mem.sliceAsBytes(xsh));
            var items: [4]kern.Item = undefined;
            var alone: [4]cuda.DeviceBuffer = undefined;
            var fused: [4]cuda.DeviceBuffer = undefined;
            for (set, 0..) |nk, i| {
                alone[i] = try cuda.DeviceBuffer.alloc(gpu.d, rows * nk[0] * 2);
                fused[i] = try cuda.DeviceBuffer.alloc(gpu.d, rows * nk[0] * 2);
                items[i] = .{ .q = weights[i], .out = fused[i].ptr };
            }
            defer for (set, 0..) |_, i| {
                alone[i].free();
                fused[i].free();
            };
            for (set, 0..) |_, i| try ops.dense(x.ptr, xs.ptr, weights[i], alone[i].ptr, rows);
            try ops.denseGroup(x.ptr, xs.ptr, items[0..set.len], rows);
            try stream.synchronize();
            for (set, 0..) |nk, i| {
                const a = try check.download(gpu, alone[i]);
                defer gpa.free(a);
                const f = try check.download(gpu, fused[i]);
                defer gpa.free(f);
                var what: [64]u8 = undefined;
                check.sameBytes(try std.fmt.bufPrint(&what, "set {d} part {d} (n {d}) rows {d}", .{ si, i, nk[0], rows }), f, a) catch {
                    failed = true;
                };
            }
        }
        check.pass("set {d}: {d} projections fused, {d} row counts", .{ si, set.len, row_counts.len });
    }
    if (failed) return error.TestFailed;
}
