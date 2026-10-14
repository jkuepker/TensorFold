//! A float64 tree attention from the bf16 bytes: output, per-slot log-sum-exp and o / l, with their bounds.

const std = @import("std");
const f64t = @import("qwen_f64.zig");

pub const HK = 2;
pub const D = 256;
pub const G = 8; // query heads a key head
pub const kv_row = HK * D;
pub const chunk = 512;
pub const group_chunks = 4;
pub const max_slots = 264; // 256: the longest unfolded stream (131071 keys, 1 row: its 63 groups stay unfolded)
pub const scale: f64 = 0.0625;
// fp32 scores from 16 chained mma steps, each rounding at 2^-23 of what it sums
const dot_eps: f64 = 16 * 0x1p-23;
const u = f64t.f32_eps; // one fp32 rounding, 2^-24
// a chunk's exp sum, relative: p 5u (ex2.approx 4u, its argument 1u), tile sum 7u, 8-tile fma chain 7u, 7 rescales x 5u
const chunk_sum: f64 = (5 + 7 + 7 + 35) * u;
// a chunk's o, relative to its l: 32 mma steps x 2u, 7 rescales x 6u (alpha 5u, its rounding 1u)
const chunk_acc: f64 = (64 + 42) * u;
// each further chunk folded into a group scales the old state by a: exp 5u, rounding 1u
const fold_step: f64 = 6 * u;
// merge: a slot's weight meets one scaling (6u) per fold after it, a convex mix moves by twice the worst weight error
const merge_step: f64 = 12 * u;

fn lseSum(chunks: usize) f64 {
    return chunk_sum + fold_step * @as(f64, @floatFromInt(chunks - 1));
}

fn oAcc(chunks: usize) f64 {
    return chunk_acc + fold_step * @as(f64, @floatFromInt(chunks - 1));
}

/// A stream's keys and values in float64 ([position][key head][dim]) with the window's nodes after the committed rows.
pub const Stream = struct {
    p: usize,
    w: usize,
    start: usize,
    groups: usize,
    parents: []const i32,
    k: []f64, // p committed rows, then the w window nodes
    v: []f64,
    kmax: [HK][D]f64, // largest |k| and |v| a column holds over every row above
    vmax: [HK][D]f64,

    pub fn build(gpa: std.mem.Allocator, p: usize, w: usize, start: usize, groups: usize, parents: []const i32, kc: []const u8, vc: []const u8, kn: []const u8, vn: []const u8) !Stream {
        const k = try gpa.alloc(f64, (p + w) * kv_row);
        const v = try gpa.alloc(f64, (p + w) * kv_row);
        for (0..p * kv_row) |i| {
            k[i] = f64t.bfAt(kc, i);
            v[i] = f64t.bfAt(vc, i);
        }
        for (0..w * kv_row) |i| {
            k[p * kv_row + i] = f64t.bfAt(kn, start * kv_row + i);
            v[p * kv_row + i] = f64t.bfAt(vn, start * kv_row + i);
        }
        var s: Stream = .{ .p = p, .w = w, .start = start, .groups = groups, .parents = parents, .k = k, .v = v, .kmax = undefined, .vmax = undefined };
        @memset(std.mem.asBytes(&s.kmax), 0);
        @memset(std.mem.asBytes(&s.vmax), 0);
        for (0..p + w) |i| for (0..HK) |hk| for (0..D) |d| { // row order: the arrays are far past the cache
            s.kmax[hk][d] = @max(s.kmax[hk][d], @abs(k[i * kv_row + hk * D + d]));
            s.vmax[hk][d] = @max(s.vmax[hk][d], @abs(v[i * kv_row + hk * D + d]));
        };
        return s;
    }

    pub fn deinit(s: *Stream, gpa: std.mem.Allocator) void {
        gpa.free(s.k);
        gpa.free(s.v);
    }

    /// The partial slot a key lands in: one per folded group, then one per chunk.
    pub fn slotOf(s: *const Stream, key: usize) usize {
        const c = key / chunk;
        return if (c < group_chunks * s.groups) c / group_chunks else c - (group_chunks - 1) * s.groups;
    }
};

pub const Row = struct {
    out: [D]f64,
    out_bound: [D]f64,
    floor: [D]f64,
    slots: usize,
    folded: usize, // the leading slots that hold a 4-chunk group; the rest hold one chunk
    live: [max_slots]bool, // the slot holds a key for this row
    lse: [max_slots]f64,
    lse_bound: [2]f64, // by slot class: a chunk, a folded group
    onorm: [max_slots][D]f64,
    onorm_bound: [2][D]f64,
    l: [max_slots]f64, // the exp sums and sums of p v per slot, kept here: too big for the stack
    o: [max_slots][D]f64,

    pub fn class(r: *const Row, slot: usize) usize {
        return @intFromBool(slot < r.folded);
    }
};

/// Row `j` of `s`, query head `head` (query `q`, D bf16 values): keys 0..p and the nodes on j's root path.
pub fn row(s: *const Stream, j: usize, head: usize, q: []const u8, scores: []f64, r: *Row) void {
    const hk = head / G;
    var path: [128]usize = undefined;
    var depth: usize = 0;
    var at: i32 = @intCast(j);
    while (at >= 0) : (at = s.parents[@intCast(at)]) {
        path[depth] = @intCast(at);
        depth += 1;
    }
    std.mem.reverse(usize, path[0..depth]);
    var qd: [D]f64 = undefined;
    for (&qd, 0..) |*x, i| x.* = f64t.bfAt(q, i);
    const nk = s.p + depth;
    var top = -std.math.inf(f64);
    for (0..nk) |t| {
        const key = if (t < s.p) t else s.p + path[t - s.p];
        const kr = s.k[key * kv_row + hk * D ..][0..D];
        var acc: [4]f64 = @splat(0);
        var d: usize = 0;
        while (d < D) : (d += 4) {
            inline for (0..4) |ui| acc[ui] += qd[d + ui] * kr[d + ui];
        }
        scores[t] = ((acc[0] + acc[1]) + (acc[2] + acc[3])) * scale;
        top = @max(top, scores[t]);
    }
    r.slots = s.slotOf(nk - 1) + 1;
    r.folded = s.groups;
    @memset(&r.live, false);
    @memset(r.l[0..r.slots], 0);
    for (r.o[0..r.slots]) |*x| @memset(x, 0);
    for (0..nk) |t| {
        const key = if (t < s.p) t else s.p + path[t - s.p];
        const slot = s.slotOf(t);
        const e = @exp(scores[t] - top);
        const vr = s.v[key * kv_row + hk * D ..][0..D];
        r.l[slot] += e;
        r.live[slot] = true;
        for (&r.o[slot], vr) |*x, y| x.* += e * y;
    }
    var total: f64 = 0;
    for (r.l[0..r.slots]) |x| total += x;
    var qk: f64 = 0; // sum |q| kmax: the largest condition of any score
    for (qd, s.kmax[hk]) |a, b| qk += @abs(a) * b;
    const delta = dot_eps * scale * qk;
    r.lse_bound = .{ delta + lseSum(1), delta + lseSum(group_chunks) };
    // the merge folds the groups, then the chunks four at a time into a running group, each group into the total
    const folds: f64 = @floatFromInt(3 + s.groups + (r.slots - s.groups + group_chunks - 1) / group_chunks);
    for (0..D) |d| {
        var sum: f64 = 0;
        for (r.o[0..r.slots]) |x| sum += x[d];
        r.out[d] = sum / total;
        const big = s.vmax[hk][d];
        // three bf16 half-ulps (rounded p, unrounded l, bf16 store), then the scores' own error and the fp32 sums
        const acc = oAcc(if (s.groups > 0) group_chunks else 1) + merge_step * folds + u;
        r.out_bound[d] = big * (3 * 0x1p-8 + 2 * delta + acc) + 0x1p-133;
        r.onorm_bound[0][d] = big * (2 * 0x1p-8 + 2 * delta + oAcc(1)) + 0x1p-133;
        r.onorm_bound[1][d] = big * (2 * 0x1p-8 + 2 * delta + oAcc(group_chunks)) + 0x1p-133;
        r.floor[d] = big * 0x1p-7;
    }
    for (0..r.slots) |slot| {
        if (!r.live[slot]) continue;
        r.lse[slot] = top + @log(r.l[slot]);
        for (&r.onorm[slot], r.o[slot]) |*x, y| x.* = y / r.l[slot];
    }
}

fn fill(rng: std.Random, bytes: []u8) void {
    for (0..bytes.len / 2) |i| std.mem.writeInt(u16, bytes[2 * i ..][0..2], f64t.bf16Of(rng.floatNorm(f32)), .little);
}

test "folded and unfolded rows match a plain softmax and keep their slots apart" {
    const gpa = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    const rng = prng.random();
    const row_ptr = try gpa.create(Row);
    defer gpa.destroy(row_ptr);
    const scores = try gpa.alloc(f64, 4096 + 128);
    defer gpa.free(scores);
    for ([_]struct { p: usize, w: usize, groups: usize }{ .{ .p = 4096, .w = 32, .groups = 2 }, .{ .p = 4096, .w = 1, .groups = 0 } }) |c| {
        const kc = try gpa.alloc(u8, c.p * kv_row * 2);
        defer gpa.free(kc);
        const vc = try gpa.alloc(u8, c.p * kv_row * 2);
        defer gpa.free(vc);
        const kn = try gpa.alloc(u8, c.w * kv_row * 2);
        defer gpa.free(kn);
        const vn = try gpa.alloc(u8, c.w * kv_row * 2);
        defer gpa.free(vn);
        var q: [D * 2]u8 = undefined;
        for ([_][]u8{ kc, vc, kn, vn, &q }) |b| fill(rng, b);
        var par: [32]i32 = undefined;
        for (&par, 0..) |*x, i| x.* = @as(i32, @intCast(i)) - 1;
        var s = try Stream.build(gpa, c.p, c.w, 0, c.groups, par[0..c.w], kc, vc, kn, vn);
        defer s.deinit(gpa);
        row(&s, c.w - 1, 3, &q, scores, row_ptr);
        const groups = c.groups;
        try std.testing.expectEqual(groups + (c.p + c.w + chunk - 1) / chunk - groups * group_chunks, row_ptr.slots);
        try std.testing.expectEqual(groups, row_ptr.folded);
        var top = -std.math.inf(f64);
        for (scores[0 .. c.p + c.w]) |x| top = @max(top, x);
        var den: f64 = 0;
        var num: f64 = 0;
        for (scores[0 .. c.p + c.w], 0..) |x, t| {
            den += @exp(x - top);
            num += @exp(x - top) * s.v[t * kv_row + (3 / G) * D + 5];
        }
        try std.testing.expectApproxEqRel(num / den, row_ptr.out[5], 1e-12);
        for (0..row_ptr.slots) |slot| try std.testing.expect(row_ptr.live[slot]);
        try std.testing.expect(row_ptr.lse_bound[1] > row_ptr.lse_bound[0]);
        try std.testing.expect(row_ptr.onorm_bound[1][5] > row_ptr.onorm_bound[0][5]);
    }
}
