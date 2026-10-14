//! Host references of add_rmsnorm with a residual: the kernel's rounded sum and the limit holding an output to it.
const std = @import("std");
const q = @import("qwen_f64.zig");

fn f32Of(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

/// h = bf16(fp32(x) + fp32(r)), the kernel's fadd_rn then round to nearest even, as float64: what it stores and norms.
pub fn sumH(x: u16, r: u16) f64 {
    return q.bf(q.bf16Of(f32Of(x) + f32Of(r)));
}

/// What an output that must equal `w` (a bf16 value) is held to: below every spacing, far above the float64 ulp.
pub fn exactLimit(w: f64) f64 {
    return q.halfUlp(w) / 4;
}

fn randomBits(rng: std.Random, mag: f32) u16 {
    return q.bf16Of((rng.float(f32) * 2 - 1) * mag);
}

test "a limit of a quarter half ulp is exact and survives the control's push" {
    var s = try q.Stat.init(std.testing.allocator, "h", 2, false);
    defer s.deinit();
    for ([_]f64{ 1.0, -1.0, 0.0, 0x1p-130, 3.140625, 0x1p100 }) |w| {
        s.add(0, w, w, exactLimit(w), 1e-30);
    }
    try std.testing.expectEqual(@as(usize, 0), s.fails);
    // one bf16 step off, up or down, from any value
    for ([_]u16{ 0x3f80, 0xbf80, 0x0001, 0x4049, 0x7f00, 0x0080 }) |b| {
        const w = q.bf(b);
        s.add(0, q.bf(b + 1), w, exactLimit(w), 1e-30);
        s.add(0, q.bf(b - 1), w, exactLimit(w), 1e-30);
    }
    try std.testing.expectEqual(@as(usize, 12), s.fails);
    // the control moves an exact match off by 1.25 limits, which float64 still sees
    var c = try q.Stat.init(std.testing.allocator, "h", 2, false);
    defer c.deinit();
    c.perturb(1);
    for ([_]f64{ 1.0, 0.0, 0x1p-130, 3.140625, 0x1p100 }) |w| c.add(1, w, w, exactLimit(w), 1e-30);
    try std.testing.expectEqual(@as(usize, 5), c.fails);
}

test "the rounded sum is within half an ulp of the exact sum, double rounding included" {
    var prng = std.Random.DefaultPrng.init(0x5eed);
    const rng = prng.random();
    for (0..200000) |_| {
        const x = randomBits(rng, @exp2(rng.float(f32) * 40 - 20));
        const r = randomBits(rng, @exp2(rng.float(f32) * 40 - 20));
        const exact = q.bf(x) + q.bf(r);
        try std.testing.expect(@abs(sumH(x, r) - exact) <= q.bound(exact, 0, 0));
    }
}

test "a kernel that rounds H toward zero makes Y exceed its tight bound with no help from Triton" {
    var prng = std.Random.DefaultPrng.init(0xa11);
    const rng = prng.random();
    const n = 512;
    const rel_norm: f64 = 0x1p-19;
    var h_rne: [n]f64 = undefined;
    var h_trunc: [n]f64 = undefined;
    var over_rne: usize = 0;
    var over_trunc: usize = 0;
    for (0..8) |_| {
        const mag = @exp2(rng.float(f32) * 4 - 2);
        var ss_rne: f64 = 0;
        var ss_trunc: f64 = 0;
        for (&h_rne, &h_trunc) |*a, *b| {
            const x = randomBits(rng, mag);
            const r = randomBits(rng, mag);
            const sum = f32Of(x) + f32Of(r);
            a.* = q.bf(q.bf16Of(sum));
            b.* = q.bf(q.bf16Trunc(sum));
            ss_rne += a.* * a.*;
            ss_trunc += b.* * b.*;
        }
        const inv_rne = 1 / @sqrt(ss_rne / n + 1e-6);
        const inv_trunc = 1 / @sqrt(ss_trunc / n + 1e-6);
        for (h_rne, h_trunc) |a, b| {
            const want = a * inv_rne;
            // a correct kernel stores the rounded want; the truncating one norms its truncated H
            over_rne += @intFromBool(@abs(q.round(want) - want) > q.bound(want, rel_norm, 0));
            over_trunc += @intFromBool(@abs(q.round(b * inv_trunc) - want) > q.bound(want, rel_norm, 0));
        }
    }
    try std.testing.expectEqual(@as(usize, 0), over_rne);
    try std.testing.expect(over_trunc > 8 * n / 4);
}
