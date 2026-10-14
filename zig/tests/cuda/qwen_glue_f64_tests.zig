//! Qwen3.6's row glue against float64: worst per-column errors of the built-in kernels and of the captured Triton's.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const qwen27 = @import("qwen27");
const check = @import("check.zig");
const base = @import("qwen_glue_tests.zig");
const st = @import("qwen_glue_f64_stats.zig");

const nref = core.norms_ref;
const aref = core.act_ref;
const glue = qwen27.glue_native;
const aot = cuda.aot;
const Rig = base.Rig;
const Out = st.Out;
const Impl = st.Impl;

const seed: u64 = 0x64f6_4a11;
const eps: f32 = 1e-6;
const hidden = glue.hidden;
const vh = glue.value_heads;
const dv = glue.value_dim;
const heads = glue.query_heads;
const hd = glue.head_dim;
const vocab = 4096;
const kdim = 2048;
// Row counts: decode, lane rows, tree windows, prompt chunks and remainders (the differential test runs more).
const row_counts = [_]usize{ 1, 2, 3, 4, 8, 16, 17, 32, 64, 128, 299, 1024, 2361 };
// A row whose sum of squares passes this overflows fp32 in every implementation: not a precision case.
const overflow: f64 = 1e38;
// Realistic rows (normal, outliers, wide) or, as stress, every kind of the differential rig but NaN and inf.
var realistic_only = true;
// Relative slack of a norm (sum tree, sqrt and division approximations, products) and of a sigmoid (ex2, division).
const rel_norm: f64 = 0x1p-19;
const rel_sigmoid: f64 = 0x1p-20;
// The sigmoid's exp argument is rounded in fp32: a relative error of |z| 2^-24 or so in exp(z),, for z below 0.
const rel_arg: f64 = 0x1p-22;

fn reps(rows: usize) usize {
    return if (rows <= 16) 6 else if (rows <= 300) 2 else 1;
}

fn impls(tri: bool) usize {
    return if (tri) 2 else 1;
}

fn f(v: u16) f64 {
    return base.bf32(v);
}

/// The sigmoid kernels' relative slack at z: the arithmetic's own, and the exp argument's for z below 0.
fn sig(rel: f64, z: f64) f64 {
    return rel + (if (z < 0) @min(-z, 100) * rel_arg else 0);
}

/// `count` rows of `width` bf16 values, each its own kind and magnitude as the differential rig (no NaN, inf).
fn matrix(r: *Rig, a: std.mem.Allocator, count: usize, width: usize, scale: f32) ![]u16 {
    const buf = try a.alloc(u16, count * width);
    for (0..count) |i| {
        var kind = r.pickKind();
        while (kind == .special or (realistic_only and kind != .normal and kind != .outliers and kind != .wide)) kind = r.pickKind();
        r.fillRow(buf[i * width ..][0..width], kind, scale);
    }
    return buf;
}

fn fetch(comptime T: type, r: *Rig, a: std.mem.Allocator, b: cuda.DeviceBuffer, n: usize) ![]T {
    try r.s.synchronize();
    const raw = try check.download(r.gpu, b);
    defer r.gpu.gpa.free(raw);
    const o = try a.alloc(T, n);
    @memcpy(std.mem.sliceAsBytes(o), raw[0 .. n * @sizeOf(T)]);
    return o;
}

fn named(a: std.mem.Allocator, tag: []const u8, what: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "{s}.{s}", .{ tag, what });
}

/// One kernel's outputs: its rows, and its fused group sums against float64 and against the rows it stored.
const Group = struct {
    h: Out,
    y: Out,
    xs: Out,
    sto: Out,

    fn init(a: std.mem.Allocator, mode: []const u8, kernel: []const u8, main: []const u8, cols: usize, groups: usize, poke: bool) !Group {
        const tag = try std.fmt.allocPrint(a, "{s}:{s}", .{ mode, kernel });
        var g: Group = .{
            .h = try Out.init(a, try named(a, tag, "H"), cols),
            .y = try Out.init(a, try named(a, tag, main), cols),
            .xs = try Out.init(a, try named(a, tag, "XS"), groups),
            .sto = try Out.init(a, try named(a, tag, "XS(stored)"), groups),
        };
        if (poke) for ([_]*Out{ &g.h, &g.y, &g.xs, &g.sto }) |o| {
            o.arm(1, 5);
        };
        return g;
    }

    fn all(g: *Group) [4]*Out {
        return .{ &g.h, &g.y, &g.xs, &g.sto };
    }

    fn report(g: *Group, triton: bool) usize {
        var bad: usize = 0;
        for (g.all()) |o| if (o.acc[0].count > 0) {
            bad += o.report(triton);
        };
        return bad;
    }

    fn failures(g: *Group, triton: bool) usize {
        var bad: usize = 0;
        for (g.all()) |o| bad += o.failures(triton);
        return bad;
    }
};

/// A fused kernel's group sums of `want` (its float64 rows), against the reference and against the rows it stored.
fn xsRow(g: *Group, im: Impl, row: usize, col0: usize, got: []const f32, want: []const f64, lim: []const f64, stored: []const f64) void {
    var ref: [32]f64 = undefined;
    var sto: [32]f64 = undefined;
    nref.groupSums(want, ref[0..got.len]);
    nref.groupSums(stored, sto[0..got.len]);
    for (got, 0..) |x, j| {
        var sl: f64 = 0;
        var sw: f64 = 0;
        var ss: f64 = 0;
        for (lim[64 * j ..][0..64], want[64 * j ..][0..64], stored[64 * j ..][0..64]) |l, w, t| {
            sl += l;
            sw += @abs(w);
            ss += @abs(t);
        }
        g.xs.note(im, row, col0 + j, x, ref[j], sl + nref.groupSumTolerance(want[64 * j ..][0..64]), sw);
        g.sto.note(im, row, col0 + j, x, sto[j], nref.groupSumTolerance(stored[64 * j ..][0..64]), ss);
    }
}

const Run = struct { h: []u16 = &.{}, y: []u16, xs: []f32 };

fn runAddRms(r: *Rig, a: std.mem.Allocator, im: Impl, x: u64, rp: ?u64, w: u64, rows: usize) !Run {
    const h = try r.out(rows * hidden * 2);
    const y = try r.out(rows * hidden * 2);
    const s = try r.out(rows * hidden / 64 * 4);
    switch (im) {
        .native => try r.native.addRmsnorm(r.s, x, rp, w, h.ptr, y.ptr, s.ptr, rows, eps),
        .triton => try r.replay.addRmsnorm(x, rp, w, h.ptr, y.ptr, s.ptr, rows, hidden, eps),
    }
    return .{ .h = try fetch(u16, r, a, h, rows * hidden), .y = try fetch(u16, r, a, y, rows * hidden), .xs = try fetch(f32, r, a, s, rows * hidden / 64) };
}

/// add_rmsnorm on random rows (or `real` ones as x): H against x + r, Y against rms(x + r) w, from the exact sum.
fn addRms(r: *Rig, g: *Group, rows_in: usize, has_r: bool, tri: bool, real: ?[]const u16) !void {
    var arena = std.heap.ArenaAllocator.init(r.gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    defer r.release();
    const rows = if (real) |x| x.len / hidden else rows_in;
    const w16 = try r.weights(hidden);
    defer r.gpu.gpa.free(w16);
    const x = real orelse try matrix(r, a, rows, hidden, 1.0);
    const res = if (has_r) try matrix(r, a, rows, hidden, 1.0) else x;
    const xb = try r.up(base.bytes(x), true);
    const rb = try r.up(base.bytes(res), true);
    const wb = try r.up(base.bytes(w16), true);
    const rp: ?u64 = if (has_r) rb.ptr else null;
    var runs: [2]Run = undefined;
    runs[0] = try runAddRms(r, a, .native, xb.ptr, rp, wb.ptr, rows);
    if (tri) runs[1] = try runAddRms(r, a, .triton, xb.ptr, rp, wb.ptr, rows);
    const w64 = try a.alloc(f64, hidden);
    for (w64, w16) |*o, v| o.* = f(v);
    const h_ref = try a.alloc(f64, hidden);
    const y_ref = try a.alloc(f64, hidden);
    const lim = try a.alloc(f64, hidden);
    const sto = try a.alloc(f64, hidden);
    const rel_y = if (has_r) 0x1p-7 + rel_norm else rel_norm; // H rounds each x + r by 2^-8 and moves 1/rms by as much
    for (0..rows) |row| {
        var ss: f64 = 0;
        for (h_ref, 0..) |*o, i| {
            const at = row * hidden + i;
            o.* = f(x[at]) + (if (has_r) f(res[at]) else 0);
            ss += o.* * o.*;
            if (has_r) for (runs[0..impls(tri)], 0..) |rn, k| {
                g.h.note(@fromBackingInt(@intCast(k)), row, i, f(rn.h[at]), o.*, st.bound(o.*, 0x1p-23, 0), @abs(o.*));
            };
        }
        if (!(ss < overflow)) {
            g.y.excluded += 1;
            continue;
        }
        nref.rms(h_ref, w64, eps, y_ref);
        for (lim, y_ref) |*l, v| l.* = st.bound(v, rel_y, 0);
        for (runs[0..impls(tri)], 0..) |rn, k| {
            const im: Impl = @fromBackingInt(@intCast(k));
            for (sto, 0..) |*o, i| {
                o.* = f(rn.y[row * hidden + i]);
                g.y.note(im, row, i, o.*, y_ref[i], lim[i], @abs(y_ref[i]));
            }
            xsRow(g, im, row, 0, rn.xs[row * hidden / 64 ..][0 .. hidden / 64], y_ref, lim, sto);
        }
    }
}

fn runGated(r: *Rig, a: std.mem.Allocator, im: Impl, p: [3]u64, rows: usize) !Run {
    const n = rows * vh * dv;
    const o = try r.out(n * 2);
    const s = try r.out(n / 64 * 4);
    switch (im) {
        .native => try r.native.gatedNorm(r.s, p[0], p[1], p[2], o.ptr, s.ptr, rows, vh, eps),
        .triton => try r.replay.gatedNorm(p[0], p[1], p[2], o.ptr, s.ptr, rows, vh, dv, eps),
    }
    return .{ .y = try fetch(u16, r, a, o, n), .xs = try fetch(f32, r, a, s, n / 64) };
}

/// gated_norm: silu(z) rms(y) w per value head, with the float64 silu and rms.
fn gated(r: *Rig, g: *Group, rows: usize, tri: bool) !void {
    var arena = std.heap.ArenaAllocator.init(r.gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    defer r.release();
    const w16 = try r.weights(dv);
    defer r.gpu.gpa.free(w16);
    const y = try matrix(r, a, rows * vh, dv, 1.0);
    const z = try matrix(r, a, rows * vh, dv, 1.5);
    const yb = try r.up(base.bytes(y), true);
    const zb = try r.up(base.bytes(z), true);
    const wb = try r.up(base.bytes(w16), true);
    var runs: [2]Run = undefined;
    runs[0] = try runGated(r, a, .native, .{ yb.ptr, zb.ptr, wb.ptr }, rows);
    if (tri) runs[1] = try runGated(r, a, .triton, .{ yb.ptr, zb.ptr, wb.ptr }, rows);
    const w64 = try a.alloc(f64, dv);
    for (w64, w16) |*o, v| o.* = f(v);
    var yv: [dv]f64 = undefined;
    var nrm: [dv]f64 = undefined;
    var want: [dv]f64 = undefined;
    var lim: [dv]f64 = undefined;
    var flo: [dv]f64 = undefined;
    var sto: [dv]f64 = undefined;
    for (0..rows * vh) |vec| {
        var ss: f64 = 0;
        for (&yv, 0..) |*o, e| {
            o.* = f(y[vec * dv + e]);
            ss += o.* * o.*;
        }
        if (!(ss < overflow)) {
            g.y.excluded += 1;
            continue;
        }
        nref.rms(&yv, w64, eps, &nrm);
        for (&want, &lim, nrm, 0..) |*o, *l, n, e| {
            const zz = f(z[vec * dv + e]);
            const silu = zz * aref.sigmoid(zz);
            o.* = n * silu;
            flo[e] = @abs(n * zz) * st.flush;
            l.* = st.bound(o.*, sig(rel_norm, zz), flo[e] + @abs(silu) * st.subnormal);
        }
        const col = vec % vh * dv;
        for (runs[0..impls(tri)], 0..) |rn, k| {
            const im: Impl = @fromBackingInt(@intCast(k));
            for (&sto, 0..) |*o, e| {
                o.* = f(rn.y[vec * dv + e]);
                g.y.note(im, vec / vh, col + e, o.*, want[e], lim[e], @max(@abs(want[e]), flo[e]));
            }
            xsRow(g, im, vec / vh, vec % vh * (dv / 64), rn.xs[vec * (dv / 64) ..][0 .. dv / 64], &want, &lim, &sto);
        }
    }
}

fn runGateMul(r: *Rig, a: std.mem.Allocator, im: Impl, o: u64, qg: u64, rows: usize) !Run {
    const n = rows * heads * hd;
    const out = try r.out(n * 2);
    const s = try r.out(n / 64 * 4);
    switch (im) {
        .native => try r.native.gateMul(r.s, o, qg, out.ptr, s.ptr, rows, heads),
        .triton => try r.replay.gateMul(o, qg, out.ptr, s.ptr, rows, heads, hd),
    }
    return .{ .y = try fetch(u16, r, a, out, n), .xs = try fetch(f32, r, a, s, n / 64) };
}

/// gate_mul: o sigmoid(gate) per head, the gate being the second half of each [q | gate] row pair.
fn gateMul(r: *Rig, g: *Group, rows: usize, tri: bool) !void {
    var arena = std.heap.ArenaAllocator.init(r.gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    defer r.release();
    const o = try matrix(r, a, rows * heads, hd, 1.0);
    const qg = try matrix(r, a, rows * heads * 2, hd, 1.5);
    const ob = try r.up(base.bytes(o), true);
    const qb = try r.up(base.bytes(qg), true);
    var runs: [2]Run = undefined;
    runs[0] = try runGateMul(r, a, .native, ob.ptr, qb.ptr, rows);
    if (tri) runs[1] = try runGateMul(r, a, .triton, ob.ptr, qb.ptr, rows);
    var want: [hd]f64 = undefined;
    var lim: [hd]f64 = undefined;
    var flo: [hd]f64 = undefined;
    var sto: [hd]f64 = undefined;
    for (0..rows * heads) |vec| {
        for (&want, &lim, 0..) |*w, *l, e| {
            const gate = f(qg[(vec * 2 + 1) * hd + e]);
            const ov = f(o[vec * hd + e]);
            w.* = ov * aref.sigmoid(gate);
            flo[e] = @abs(ov) * st.flush;
            l.* = st.bound(w.*, sig(rel_sigmoid, gate), flo[e]);
        }
        const col = vec % heads * hd;
        for (runs[0..impls(tri)], 0..) |rn, k| {
            const im: Impl = @fromBackingInt(@intCast(k));
            for (&sto, 0..) |*s, e| {
                s.* = f(rn.y[vec * hd + e]);
                g.y.note(im, vec / heads, col + e, s.*, want[e], lim[e], @max(@abs(want[e]), flo[e]));
            }
            xsRow(g, im, vec / heads, vec % heads * (hd / 64), rn.xs[vec * (hd / 64) ..][0 .. hd / 64], &want, &lim, &sto);
        }
    }
}

/// group_sums of `rows` rows of `x` (stride `ld`, `k` values): fp32 sums against float64 sums; `dump` is Triton's.
fn sums(r: *Rig, o: *Out, x: []const u16, ld: usize, rows: usize, k: usize, tri: bool, dump: ?[]const f32) !void {
    var arena = std.heap.ArenaAllocator.init(r.gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    defer r.release();
    const kg = k / 64;
    const xb = try r.up(base.bytes(x), true);
    const sn = try r.out(rows * kg * 4);
    try r.native.groupSums(r.s, xb.ptr, sn.ptr, ld, rows, k);
    const nat = try fetch(f32, r, a, sn, rows * kg);
    var other: ?[]const f32 = dump;
    if (dump == null and tri) {
        const stt = try r.out(rows * kg * 4);
        try r.replay.groupSums(xb.ptr, stt.ptr, ld, rows, k);
        other = try fetch(f32, r, a, stt, rows * kg);
    }
    const v = try a.alloc(f64, k);
    const ref = try a.alloc(f64, kg);
    for (0..rows) |row| {
        for (v, 0..) |*e, i| e.* = f(x[row * ld + i]);
        nref.groupSums(v, ref);
        for (ref, 0..) |want, j| {
            const lim = nref.groupSumTolerance(v[64 * j ..][0..64]);
            var scale: f64 = 0;
            for (v[64 * j ..][0..64]) |e| scale += @abs(e);
            o.note(.native, row, j, nat[row * kg + j], want, lim, scale);
            if (other) |t| o.note(.triton, row, j, t[row * kg + j], want, lim, scale);
        }
    }
}

/// The 4-bit affine embedding table of the differential test: random words, scales near 0.05, biases within 1.
const Table = struct {
    words: []u32,
    sc: []u16,
    bi: []u16,
    dev: [3]cuda.DeviceBuffer,

    fn init(r: *Rig, a: std.mem.Allocator) !Table {
        const rng = r.rand();
        var t: Table = .{
            .words = try a.alloc(u32, vocab * kdim / 8),
            .sc = try a.alloc(u16, vocab * kdim / 64),
            .bi = try a.alloc(u16, vocab * kdim / 64),
            .dev = undefined,
        };
        for (t.words) |*w| w.* = rng.int(u32);
        for (t.sc) |*o| o.* = base.bf((rng.float(f32) + 0.05) * 0.1);
        for (t.bi) |*o| o.* = base.bf((rng.float(f32) - 0.5) * 2.0);
        t.dev[0] = try cuda.DeviceBuffer.fromHost(r.gpu.d, std.mem.sliceAsBytes(t.words));
        t.dev[1] = try cuda.DeviceBuffer.fromHost(r.gpu.d, base.bytes(t.sc));
        t.dev[2] = try cuda.DeviceBuffer.fromHost(r.gpu.d, base.bytes(t.bi));
        return t;
    }

    fn deinit(t: *Table) void {
        for (&t.dev) |*b| b.free();
    }
};

/// embed: q scale + bias in float64 from the same words, scales and biases, for random token rows.
fn embed(r: *Rig, o: *Out, t: Table, rows: usize, tri: bool) !void {
    var arena = std.heap.ArenaAllocator.init(r.gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    defer r.release();
    const rng = r.rand();
    const ids = try a.alloc(i32, rows);
    for (ids) |*x| x.* = @intCast(rng.uintLessThan(u32, vocab));
    ids[0] = if (rng.boolean()) 0 else vocab - 1;
    const ib = try r.up(std.mem.sliceAsBytes(ids), true);
    const on = try r.out(rows * kdim * 2);
    try r.native.embed(r.s, ib.ptr, t.dev[0].ptr, t.dev[1].ptr, t.dev[2].ptr, on.ptr, rows, kdim);
    const nat = try fetch(u16, r, a, on, rows * kdim);
    var other: ?[]u16 = null;
    if (tri) {
        const ot = try r.out(rows * kdim * 2);
        try r.replay.embed(ib.ptr, t.dev[0].ptr, t.dev[1].ptr, t.dev[2].ptr, ot.ptr, rows, kdim);
        other = try fetch(u16, r, a, ot, rows * kdim);
    }
    for (ids, 0..) |id, row| {
        const tok: usize = @intCast(id);
        for (0..kdim) |i| {
            const word = t.words[tok * (kdim / 8) + i / 8];
            const q: u4 = @intCast((word >> @intCast(4 * (i % 8))) & 0xf);
            const want = aref.affine4(q, f(t.sc[tok * (kdim / 64) + i / 64]), f(t.bi[tok * (kdim / 64) + i / 64]));
            const lim = st.bound(want, 0x1p-23, 0);
            o.note(.native, row, i, f(nat[row * kdim + i]), want, lim, @abs(want));
            if (other) |t2| o.note(.triton, row, i, f(t2[row * kdim + i]), want, lim, @abs(want));
        }
    }
}

/// paths on the host: the chain from the node up, its last `max_depth` rows root first, -1 past the depth.
fn hostPaths(a: std.mem.Allocator, parents: []const i32, out: []i32, depths: []i32) !void {
    @memset(out, -1);
    var chain = std.ArrayList(i32).empty;
    for (parents, 0..) |_, node| {
        chain.clearRetainingCapacity();
        var cur: i32 = @intCast(node);
        while (cur >= 0) : (cur = parents[@intCast(cur)]) try chain.append(a, cur);
        const depth = @min(chain.items.len, glue.max_depth);
        depths[node] = @intCast(depth);
        for (0..depth) |slot| out[node * glue.max_depth + slot] = chain.items[depth - 1 - slot];
    }
}

/// The integer kernel: whole PATHS and DEPTHS buffers equal to the host's, no tolerance.
const Exact = struct {
    cases: usize = 0,
    bad: [2]usize = .{ 0, 0 },
    poke: bool = false,
};

fn paths(r: *Rig, e: *Exact, rows: usize, shape: usize, tri: bool) !void {
    var arena = std.heap.ArenaAllocator.init(r.gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    defer r.release();
    const parents = try a.alloc(i32, rows);
    base.forest(r, parents, shape);
    const pb = try r.up(std.mem.sliceAsBytes(parents), true);
    const want_p = try a.alloc(i32, rows * glue.max_depth);
    const want_d = try a.alloc(i32, rows);
    try hostPaths(a, parents, want_p, want_d);
    e.cases += 1;
    for (0..if (tri) 2 else 1) |k| {
        const po = try r.out(rows * glue.max_depth * 4);
        const dp = try r.out(rows * 4);
        if (k == 0) try r.native.paths(r.s, pb.ptr, po.ptr, dp.ptr, rows) else try r.replay.paths(pb.ptr, po.ptr, dp.ptr, rows);
        const got_p = try fetch(i32, r, a, po, rows * glue.max_depth);
        const got_d = try fetch(i32, r, a, dp, rows);
        if (e.poke and k == 0) got_p[(rows - 1) * glue.max_depth] +%= 1;
        if (!std.mem.eql(i32, got_p, want_p) or !std.mem.eql(i32, got_d, want_d)) e.bad[k] += 1;
    }
}

/// Every dumped y (and every eighth h) of the Python engine's steps: real rows through group_sums and add_rmsnorm.
fn realRows(r: *Rig, sum_out: *Out, add: *Group, oracle: []const u8, tri: bool) !void {
    var files: usize = 0;
    var arena = std.heap.ArenaAllocator.init(r.gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_][]const u8{ "story", "code", "facts", "long" }) |prompt| for ([_][]const u8{ "cold", "warm", "prefill" }) |phase| {
        var step: usize = 0;
        while (step < 4096) : (step += 1) {
            var dir_buf: [256]u8 = undefined;
            const dir = if (std.mem.eql(u8, phase, "prefill"))
                try std.fmt.bufPrint(&dir_buf, "{s}/{s}/{s}", .{ oracle, prompt, phase })
            else
                try std.fmt.bufPrint(&dir_buf, "{s}/{s}/{s}/step{d:0>3}", .{ oracle, prompt, phase, step });
            var n: usize = 0;
            while (n < 512) : (n += 1) {
                _ = arena.reset(.retain_capacity);
                var path_buf: [320]u8 = undefined;
                const y_path = try std.fmt.bufPrint(&path_buf, "{s}/{d:0>3}_y.bin", .{ dir, n });
                const y_raw = std.Io.Dir.cwd().readFileAlloc(r.gpu.io, y_path, a, .limited(1 << 30)) catch |e| switch (e) {
                    error.FileNotFound => break,
                    else => return e,
                };
                const rows = y_raw.len / (kdim * 2);
                if (rows == 0 or y_raw.len % (kdim * 2) != 0) return error.UnexpectedDump;
                const x = try a.alloc(u16, rows * kdim);
                @memcpy(std.mem.sliceAsBytes(x), y_raw);
                const xs_path = try std.fmt.bufPrint(&path_buf, "{s}/{d:0>3}_xs.bin", .{ dir, n });
                const xs_raw = try std.Io.Dir.cwd().readFileAlloc(r.gpu.io, xs_path, a, .limited(1 << 30));
                const dump = try a.alloc(f32, rows * kdim / 64);
                @memcpy(std.mem.sliceAsBytes(dump), xs_raw);
                try sums(r, sum_out, x, kdim, rows, kdim, false, dump);
                files += 1;
                if (files % 8 != 0) continue;
                const h_path = try std.fmt.bufPrint(&path_buf, "{s}/{d:0>3}_h.bin", .{ dir, n });
                const h_raw = try std.Io.Dir.cwd().readFileAlloc(r.gpu.io, h_path, a, .limited(1 << 30));
                const hx = try a.alloc(u16, h_raw.len / 2);
                @memcpy(std.mem.sliceAsBytes(hx), h_raw);
                try addRms(r, add, 0, false, tri, hx);
            }
            if (n == 0 or std.mem.eql(u8, phase, "prefill")) break;
        }
    };
    std.debug.print("   {d} dumped y files\n", .{files});
}

/// Every kernel's outputs; `poke` is the negative control (one native element per output moved off its reference).
const Outs = struct {
    add0: Group,
    add1: Group,
    real_add: Group,
    gated: Group,
    gate: Group,
    sums: Out,
    real_sums: Out,
    embed: Out,
    paths: Exact = .{},

    fn init(a: std.mem.Allocator, poke: bool, mode: []const u8) !Outs {
        var o: Outs = .{
            .add0 = try Group.init(a, mode, "add_rmsnorm[HAS_R=0]", "Y", hidden, hidden / 64, poke),
            .add1 = try Group.init(a, mode, "add_rmsnorm[HAS_R=1]", "Y", hidden, hidden / 64, poke),
            .real_add = try Group.init(a, mode, "add_rmsnorm[real h rows]", "Y", hidden, hidden / 64, poke),
            .gated = try Group.init(a, mode, "gated_norm", "OUT", vh * dv, vh * dv / 64, poke),
            .gate = try Group.init(a, mode, "gate_mul", "OUT", heads * hd, heads * hd / 64, poke),
            .sums = try Out.init(a, try std.fmt.allocPrint(a, "{s}:group_sums.XS", .{mode}), kdim / 64),
            .real_sums = try Out.init(a, try std.fmt.allocPrint(a, "{s}:group_sums[real y rows].XS", .{mode}), kdim / 64),
            .embed = try Out.init(a, try std.fmt.allocPrint(a, "{s}:embed.OUT", .{mode}), kdim),
        };
        if (poke) for ([_]*Out{ &o.sums, &o.real_sums, &o.embed }) |x| {
            x.arm(1, 5);
        };
        o.paths.poke = poke;
        return o;
    }

    /// The first armed output that took part in the run and still passed.
    fn missed(o: *Outs, tri: bool) ?[]const u8 {
        for ([_]*Group{ &o.add0, &o.add1, &o.gated, &o.gate }) |g| for (g.all()) |x| {
            if (x.acc[0].count > 0 and x.failures(tri) == 0) return x.name;
        };
        for ([_]*Out{ &o.sums, &o.embed }) |x| if (x.failures(tri) == 0) return x.name;
        return null;
    }

    fn report(o: *Outs, tri: bool, oracle: bool) usize {
        var bad = o.add0.report(tri) + o.add1.report(tri) + o.gated.report(tri) + o.gate.report(tri);
        bad += o.sums.report(tri);
        if (o.embed.acc[0].count > 0) bad += o.embed.report(tri);
        if (oracle) bad += o.real_sums.report(true) + o.real_add.report(tri);
        if (o.paths.cases > 0) std.debug.print("F64 paths exact cases={d} native differ={d} triton differ={d}\n", .{ o.paths.cases, o.paths.bad[0], o.paths.bad[1] });
        return bad + o.paths.bad[0] + o.paths.bad[1];
    }
};

/// The served shapes on random rows: `rounds` of every row count, the kernels run in turn.
fn sweep(r: *Rig, o: *Outs, table: Table, counts: []const usize, tri: bool, rounds: usize, stress: bool) !void {
    realistic_only = !stress;
    for (counts) |rows| for (0..@min(rounds, reps(rows))) |_| {
        try addRms(r, &o.add0, rows, false, tri, null);
        try addRms(r, &o.add1, rows, true, tri, null);
        try gated(r, &o.gated, rows, tri);
        try gateMul(r, &o.gate, rows, tri);
        if (!stress) try embed(r, &o.embed, table, rows, tri);
    };
    for ([_]usize{ kdim, kdim + 16, 2 * kdim + 48 }) |ld| for (counts) |rows| for (0..@min(rounds, reps(rows))) |_| {
        if (rows > 128) continue;
        const x = try matrix(r, r.gpu.gpa, rows, ld, 1.0);
        defer r.gpu.gpa.free(x);
        try sums(r, &o.sums, x, ld, rows, kdim, tri, null);
    };
    if (!stress) for (base.path_rows) |rows| for (0..5) |shape| for (0..@min(rounds, 2)) |_| try paths(r, &o.paths, rows, shape, tri);
}

pub fn run(gpu: check.Gpu, aot_dir: ?[]const u8, oracle: ?[]const u8) !void {
    var set: ?aot.Set = null;
    if (aot_dir) |dir| set = try aot.Set.load(gpu.gpa, gpu.io, gpu.d, gpu.ctx.device, dir);
    defer if (set) |*s| s.deinit();
    var native = try glue.Native.load(gpu.d, try gpu.ctx.capability());
    defer native.deinit();
    var r: Rig = .{ .gpu = gpu, .s = try cuda.Stream.init(gpu.d, true), .native = native, .replay = undefined, .prng = std.Random.DefaultPrng.init(seed) };
    if (set) |*s| r.replay = .{ .set = s, .s = r.s, .glue = null };
    const tri = set != null;
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var table = try Table.init(&r, a);
    defer table.deinit();
    std.debug.print("-- qwen-glue-f64: native vs float64, Triton {s}, oracle {s}\n", .{ if (tri) "replay" else "none", if (oracle != null) "dumps" else "none" });
    var o = try Outs.init(a, false, "realistic");
    try sweep(&r, &o, table, &row_counts, tri, 6, false);
    if (oracle) |dir| try realRows(&r, &o.real_sums, &o.real_add, dir, tri);
    var x = try Outs.init(a, false, "stress");
    try sweep(&r, &x, table, &row_counts, tri, 2, true);
    const bad = o.report(tri, oracle != null) + x.report(tri, false);
    // The negative control: one native element per output three bounds off must fail every output.
    var c = try Outs.init(a, true, "control");
    try sweep(&r, &c, table, &.{3}, tri, 1, false);
    const missed = c.missed(tri);
    if (missed) |name| std.debug.print("FAIL control: {s} accepted a perturbed native element\n", .{name});
    std.debug.print("   control: every output rejected its perturbed element, paths {s}\n", .{if (c.paths.bad[0] > 0) "rejected" else "accepted"});
    std.debug.print("RESULT qwen-glue-f64: {d} outside bounds or worse than Triton\n", .{bad});
    try check.expect(bad == 0 and missed == null and c.paths.bad[0] > 0, "qwen-glue-f64", .{});
}
