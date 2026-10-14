//! A float64 tree attention from the bf16 bytes: output, per-slot log-sum-exp and o / l, with their bounds.

const std = @import("std");
const f64t = @import("qwen_f64.zig");

pub const HK = 2;
pub const D = 256;
pub const G = 8; // query heads a key head
pub const kv_row = HK * D;
pub const chunk = 512;
pub const group_chunks = 4;
pub const max_slots = 40;
pub const scale: f64 = 0.0625;
// fp32 scores from 16 chained mma steps, each rounding at 2^-23 of what it sums
const dot_eps: f64 = 16 * 0x1p-23;

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
        for (&s.kmax, &s.vmax, 0..) |*km, *vm, hk| for (km, vm, 0..) |*a, *b, d| {
            a.* = 0;
            b.* = 0;
            for (0..p + w) |i| {
                a.* = @max(a.*, @abs(k[i * kv_row + hk * D + d]));
                b.* = @max(b.*, @abs(v[i * kv_row + hk * D + d]));
            }
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
    live: [max_slots]bool, // the slot holds a key for this row
    lse: [max_slots]f64,
    lse_bound: f64,
    onorm: [max_slots][D]f64,
    onorm_bound: [D]f64,
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
            inline for (0..4) |u| acc[u] += qd[d + u] * kr[d + u];
        }
        scores[t] = ((acc[0] + acc[1]) + (acc[2] + acc[3])) * scale;
        top = @max(top, scores[t]);
    }
    r.slots = s.slotOf(nk - 1) + 1;
    var l: [max_slots]f64 = @splat(0);
    @memset(&r.live, false);
    var o: [max_slots][D]f64 = undefined;
    for (o[0..r.slots]) |*x| @memset(x, 0);
    for (0..nk) |t| {
        const key = if (t < s.p) t else s.p + path[t - s.p];
        const slot = s.slotOf(t);
        const e = @exp(scores[t] - top);
        const vr = s.v[key * kv_row + hk * D ..][0..D];
        l[slot] += e;
        r.live[slot] = true;
        for (&o[slot], vr) |*x, y| x.* += e * y;
    }
    var total: f64 = 0;
    for (l[0..r.slots]) |x| total += x;
    var qk: f64 = 0; // sum |q| kmax: the largest condition of any score
    for (qd, s.kmax[hk]) |a, b| qk += @abs(a) * b;
    const delta = dot_eps * scale * qk;
    r.lse_bound = delta + 0x1p-20;
    for (0..D) |d| {
        var sum: f64 = 0;
        for (o[0..r.slots]) |x| sum += x[d];
        r.out[d] = sum / total;
        const big = s.vmax[hk][d];
        // three bf16 half-ulps (rounded p, unrounded l, bf16 store), then the scores' own error
        r.out_bound[d] = big * (3 * 0x1p-8 + 2 * delta) + 0x1p-133;
        r.onorm_bound[d] = big * (2 * 0x1p-8 + 2 * delta) + 0x1p-133;
        r.floor[d] = big * 0x1p-7;
    }
    for (0..r.slots) |slot| {
        if (!r.live[slot]) continue;
        r.lse[slot] = top + @log(l[slot]);
        for (&r.onorm[slot], o[slot]) |*x, y| x.* = y / l[slot];
    }
}
