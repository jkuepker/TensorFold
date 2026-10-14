//! Adversarial group_sums rows and the per-element-depth bound that tells the kernel's order from a worse one.
const std = @import("std");
const q = @import("qwen_f64.zig");

const u = q.f32_eps;
const thread_values = 8;
const group_values = 64;
// The first rows of a group's ladder: 7 values just over half an fp32 ulp of B, the 8th (last of the chain) being B.
const small_rel: f64 = 0x1p-24 * (1.0 + 0x1p-7);

/// fp32 adds on the path of element `i` of a group: its place in the thread's chain of 8, then 3 butterfly adds.
pub fn depthOf(i: usize) usize {
    const p = i % thread_values;
    return (if (p < 2) @as(usize, 7) else thread_values - p) + 3;
}

/// A group's sum error bound in this kernel's order: sum |x_i| gamma(d_i) with d_i the adds on x_i's path.
pub fn bound(x: []const f64) f64 {
    var t: f64 = 0;
    for (x, 0..) |v, i| {
        const d: f64 = @floatFromInt(depthOf(i));
        t += @abs(v) * d * u / (1 - d * u);
    }
    return t + 1e-30;
}

/// Fills `row` (a multiple of 64 values) with ladder groups, all else 0; `seed` varies the place, sign and scale.
pub fn fill(row: []u16, seed: usize) void {
    @memset(row, 0);
    for (0..row.len / group_values) |j| {
        const k = seed + j;
        const big: f32 = (if ((k / 3) % 2 == 0) @as(f32, 1) else -1) * @as(f32, @floatFromInt(@as(u32, 1) << @intCast(k % 5))) * 0.25;
        const small: f32 = @floatCast(@as(f64, big) * small_rel);
        // even groups: B and its smalls in one chain; odd ones: the smalls in the next thread's chain
        const t = if (k % 2 == 0) (k / 2) % thread_values else (k / 2) % (thread_values - 1);
        const at = j * group_values + t * thread_values;
        const smalls = if (k % 2 == 0) at else at + thread_values;
        for (0..thread_values - 1) |p| row[smalls + p] = q.bf16Of(small);
        row[at + thread_values - 1] = q.bf16Of(big);
    }
}

fn chain(v: []const f32, reversed: bool) f32 {
    var s: f32 = if (reversed) v[7] + v[6] else v[0] + v[1];
    for (2..thread_values) |i| s += if (reversed) v[thread_values - 1 - i] else v[i];
    return s;
}

// The kernel's reduction (a chain of 8 per thread, then butterfly 4, 2, 1) and orders that differ from it.
const Order = enum { kernel, reversed_chain, butterfly_up, pairwise_chain, sequential };

fn sum(order: Order, x: []const f32) f32 {
    if (order == .sequential) {
        var s = x[0];
        for (x[1..group_values]) |v| s += v;
        return s;
    }
    var lanes: [thread_values]f32 = undefined;
    for (&lanes, 0..) |*l, t| {
        const v = x[t * thread_values ..][0..thread_values];
        l.* = switch (order) {
            .reversed_chain => chain(v, true),
            .pairwise_chain => ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7])),
            else => chain(v, false),
        };
    }
    const offsets = if (order == .butterfly_up) [_]usize{ 1, 2, 4 } else [_]usize{ 4, 2, 1 };
    for (offsets) |off| {
        var next = lanes;
        for (&next, 0..) |*l, t| l.* = lanes[t] + lanes[t ^ off];
        lanes = next;
    }
    return lanes[0];
}

/// The order's error against the float64 sum of one group of bf16 bits.
fn errorOf(order: Order, g: []const u16) f64 {
    var x: [group_values]f32 = undefined;
    var exact: f64 = 0;
    for (g, &x) |b, *o| {
        o.* = @floatCast(q.bf(b));
        exact += o.*;
    }
    return @abs(sum(order, &x) - exact);
}

fn groupValues(g: []const u16) [group_values]f64 {
    var v: [group_values]f64 = undefined;
    for (g, &v) |b, *o| o.* = q.bf(b);
    return v;
}

fn ladder(gpa: std.mem.Allocator, seed: usize) ![]u16 {
    const r = try gpa.alloc(u16, 4 * group_values);
    fill(r, seed);
    return r;
}

test "the ladder catches a reversed chain and a sequential sum, never the kernel's order or an equal-depth one" {
    const gpa = std.testing.allocator;
    var caught: [5]usize = @splat(0);
    var seen: [2]usize = @splat(0);
    var worst_kernel: f64 = 0;
    for (0..40) |seed| {
        const r = try ladder(gpa, seed);
        defer gpa.free(r);
        for (0..4) |j| {
            const g = r[j * group_values ..][0..group_values];
            const v = groupValues(g);
            const b = bound(&v);
            seen[(seed + j) % 2] += 1;
            worst_kernel = @max(worst_kernel, errorOf(.kernel, g) / b);
            for (std.enums.values(Order), 0..) |o, i| {
                if (errorOf(o, g) > b) caught[i] += 1;
            }
        }
    }
    try std.testing.expect(worst_kernel < 1);
    // the kernel's own order is never caught; a reversed chain only where the smalls share B's chain (even groups)
    try std.testing.expectEqual(@as(usize, 0), caught[@intFromEnum(Order.kernel)]);
    try std.testing.expectEqual(seen[0], caught[@intFromEnum(Order.reversed_chain)]);
    // a sequential sum only where the smalls come after B (odd groups, the next thread's chain)
    try std.testing.expectEqual(seen[1], caught[@intFromEnum(Order.sequential)]);
    // the same depth in another order, and a shallower tree, cannot be told from the kernel's
    try std.testing.expectEqual(@as(usize, 0), caught[@intFromEnum(Order.butterfly_up)]);
    try std.testing.expectEqual(@as(usize, 0), caught[@intFromEnum(Order.pairwise_chain)]);
}

test "on the ladder the numbers the report quotes" {
    var g: [group_values]u16 = undefined;
    for ([_]usize{ 0, 1 }) |seed| {
        fill(&g, seed);
        const v = groupValues(&g);
        var big: f64 = 0;
        for (v) |e| big = @max(big, @abs(e));
        const kernel = errorOf(.kernel, &g) / (u * big);
        try std.testing.expect(kernel < 1);
        try std.testing.expectApproxEqAbs(@as(f64, 4), bound(v[0..]) / (u * big), 1e-4);
        const wrong = errorOf(if (seed == 0) .reversed_chain else .sequential, &g) / (u * big);
        try std.testing.expect(wrong > 6.5 and wrong < 7);
    }
}

test "random rows cannot tell the orders apart: each stays under the depth-10 bound" {
    var prng = std.Random.DefaultPrng.init(0x1add);
    const rng = prng.random();
    var g: [group_values]u16 = undefined;
    var over_depth: usize = 0;
    var over_each: usize = 0;
    for (0..20000) |_| {
        const mag = @exp2(rng.float(f32) * 8 - 4);
        for (&g) |*o| o.* = q.bf16Of((rng.float(f32) * 2 - 1) * mag);
        const v = groupValues(&g);
        for ([_]Order{ .reversed_chain, .butterfly_up, .pairwise_chain }) |o| {
            if (errorOf(o, &g) > q.sumBound(&v, 10)) over_depth += 1;
            if (errorOf(o, &g) > bound(&v)) over_each += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 0), over_depth);
    try std.testing.expectEqual(@as(usize, 0), over_each);
}
