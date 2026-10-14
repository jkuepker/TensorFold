//! Qwen3.6's row glue against float64: worst per-column errors of the built-in kernels and of the captured Triton's.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const base = @import("qwen_glue_tests.zig");
const q = @import("qwen_f64.zig");
const c = @import("qwen_glue_f64_common.zig");
const ladder = @import("qwen_glue_f64_ladder.zig");
const ref = @import("qwen_glue_f64_ref.zig");
const real = @import("qwen_glue_f64_real.zig");
const emb = @import("qwen_glue_f64_embed.zig");

const nref = core.norms_ref;
const aref = core.act_ref;
const glue = c.glue;
const aot = cuda.aot;
const Rig = c.Rig;
const Out = c.Out;
const Group = c.Group;
const f = c.f;
const hidden = c.hidden;
const vh = c.vh;
const dv = c.dv;
const heads = c.heads;
const hd = c.hd;
const kdim = c.kdim;

const seed: u64 = 0x64f6_4a11;
// Row counts: decode, lane rows, tree windows, prompt chunks and remainders (the differential test runs more).
const row_counts = [_]usize{ 1, 2, 3, 4, 8, 16, 17, 32, 64, 128, 299, 1024, 2361 };
// Realistic rows (normal, outliers, wide) or, as stress, every kind of the differential rig but NaN and inf.
var realistic_only = true;
// The negative control's column: every native element there moves 1.25 bounds off its reference.
const control_col = 5;

fn reps(rows: usize) usize {
    return if (rows <= 16) 6 else if (rows <= 300) 2 else 1;
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

const Run = struct { h: []u16 = &.{}, y: []u16, xs: []f32 };

fn runAddRms(r: *Rig, a: std.mem.Allocator, im: c.Impl, x: u64, rp: ?u64, w: u64, rows: usize) !Run {
    const h = try r.out(rows * hidden * 2);
    const y = try r.out(rows * hidden * 2);
    const s = try r.out(rows * hidden / 64 * 4);
    switch (im) {
        .native => try r.native.addRmsnorm(r.s, x, rp, w, h.ptr, y.ptr, s.ptr, rows, c.eps),
        .triton => try r.replay.addRmsnorm(x, rp, w, h.ptr, y.ptr, s.ptr, rows, hidden, c.eps),
    }
    return .{ .h = try c.fetch(u16, r, a, h, rows * hidden), .y = try c.fetch(u16, r, a, y, rows * hidden), .xs = try c.fetch(f32, r, a, s, rows * hidden / 64) };
}

/// add_rmsnorm on random rows (or `rows_real`): H against the host's rounded x + r, Y against the rms of that.
fn addRms(r: *Rig, g: *Group, rows_in: usize, has_r_in: bool, tri: bool, rows_real: ?c.Real) !void {
    var arena = std.heap.ArenaAllocator.init(r.gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    defer r.release();
    const rows = if (rows_real) |p| p.x.len / hidden else rows_in;
    const has_r = if (rows_real) |p| p.r != null else has_r_in;
    const w16 = try r.weights(hidden);
    defer r.gpu.gpa.free(w16);
    const x = if (rows_real) |p| p.x else try matrix(r, a, rows, hidden, 1.0);
    const res: ?[]const u16 = if (rows_real) |p| p.r else if (has_r) try matrix(r, a, rows, hidden, 1.0) else null;
    const xb = try r.up(base.bytes(x), true);
    const rp: ?u64 = if (res) |s| (try r.up(base.bytes(s), true)).ptr else null;
    const wb = try r.up(base.bytes(w16), true);
    const w64 = try a.alloc(f64, hidden);
    for (w64, w16) |*o, v| o.* = f(v);
    const exact = try a.alloc(f64, hidden);
    const h_ref = try a.alloc(f64, hidden);
    const y_ref = try a.alloc(f64, hidden);
    const lim = try a.alloc(f64, hidden);
    const sto = try a.alloc(f64, hidden);
    const sto_t = try a.alloc(f64, hidden);
    var sp: [2]usize = undefined;
    for (c.spans(rows, &sp)) |n| {
        const nat = try runAddRms(r, a, .native, xb.ptr, rp, wb.ptr, n);
        const tr: ?Run = if (tri) try runAddRms(r, a, .triton, xb.ptr, rp, wb.ptr, n) else null;
        for (0..n) |row| {
            var ss: f64 = 0;
            for (0..hidden) |i| {
                const at = row * hidden + i;
                exact[i] = f(x[at]) + (if (res) |s| f(s[at]) else 0);
                h_ref[i] = if (res) |s| ref.sumH(x[at], s[at]) else exact[i];
                ss += h_ref[i] * h_ref[i];
                if (!has_r) continue;
                const got_t: ?f64 = if (tr) |e| f(e.h[at]) else null;
                g.h.at(n).add(i, f(nat.h[at]), got_t, h_ref[i], ref.exactLimit(h_ref[i]), c.floor);
                g.hx.at(n).add(i, f(nat.h[at]), got_t, exact[i], q.bound(exact[i], 0, 0), c.floor);
            }
            if (!(ss < c.overflow)) {
                g.excluded += 1;
                continue;
            }
            nref.rms(h_ref, w64, c.eps, y_ref);
            for (lim, y_ref) |*l, v| l.* = q.bound(v, c.rel_norm, 0);
            const at = row * hidden;
            for (sto, sto_t, 0..) |*o, *t, i| {
                o.* = f(nat.y[at + i]);
                t.* = if (tr) |e| f(e.y[at + i]) else 0;
                g.y.at(n).add(i, o.*, if (tr != null) t.* else null, y_ref[i], lim[i], c.floor);
            }
            const gs = hidden / 64;
            const nat_row: c.Row = .{ .xs = nat.xs[row * gs ..][0..gs], .y = sto };
            const tri_row: ?c.Row = if (tr) |e| .{ .xs = e.xs[row * gs ..][0..gs], .y = sto_t } else null;
            c.xsRow(g, n, 0, nat_row, tri_row, y_ref, lim);
        }
    }
}

fn runGated(r: *Rig, a: std.mem.Allocator, im: c.Impl, p: [3]u64, rows: usize) !Run {
    const n = rows * vh * dv;
    const o = try r.out(n * 2);
    const s = try r.out(n / 64 * 4);
    switch (im) {
        .native => try r.native.gatedNorm(r.s, p[0], p[1], p[2], o.ptr, s.ptr, rows, vh, c.eps),
        .triton => try r.replay.gatedNorm(p[0], p[1], p[2], o.ptr, s.ptr, rows, vh, dv, c.eps),
    }
    return .{ .y = try c.fetch(u16, r, a, o, n), .xs = try c.fetch(f32, r, a, s, n / 64) };
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
    const w64 = try a.alloc(f64, dv);
    for (w64, w16) |*o, v| o.* = f(v);
    var yv: [dv]f64 = undefined;
    var nrm: [dv]f64 = undefined;
    var want: [dv]f64 = undefined;
    var lim: [dv]f64 = undefined;
    var flo: [dv]f64 = undefined;
    var sto: [dv]f64 = undefined;
    var sto_t: [dv]f64 = undefined;
    var sp: [2]usize = undefined;
    for (c.spans(rows, &sp)) |n| {
        const nat = try runGated(r, a, .native, .{ yb.ptr, zb.ptr, wb.ptr }, n);
        const tr: ?Run = if (tri) try runGated(r, a, .triton, .{ yb.ptr, zb.ptr, wb.ptr }, n) else null;
        for (0..n * vh) |vec| {
            var ss: f64 = 0;
            for (&yv, 0..) |*o, e| {
                o.* = f(y[vec * dv + e]);
                ss += o.* * o.*;
            }
            if (!(ss < c.overflow)) {
                g.excluded += 1;
                continue;
            }
            nref.rms(&yv, w64, c.eps, &nrm);
            for (&want, &lim, nrm, 0..) |*o, *l, nm, e| {
                const zz = f(z[vec * dv + e]);
                const silu = zz * aref.sigmoid(zz);
                o.* = nm * silu;
                flo[e] = @abs(nm * zz) * q.flt_min;
                l.* = q.bound(o.*, c.sig(c.rel_norm, zz), flo[e] + @abs(silu) * q.subnormal);
            }
            const col = vec % vh * dv;
            for (&sto, &sto_t, 0..) |*o, *t, e| {
                o.* = f(nat.y[vec * dv + e]);
                t.* = if (tr) |x| f(x.y[vec * dv + e]) else 0;
                g.y.at(n).add(col + e, o.*, if (tr != null) t.* else null, want[e], lim[e], @max(@abs(want[e]), flo[e], c.floor));
            }
            const gs = dv / 64;
            const nat_row: c.Row = .{ .xs = nat.xs[vec * gs ..][0..gs], .y = &sto };
            const tri_row: ?c.Row = if (tr) |x| .{ .xs = x.xs[vec * gs ..][0..gs], .y = &sto_t } else null;
            c.xsRow(g, n, vec % vh * gs, nat_row, tri_row, &want, &lim);
        }
    }
}

fn runGateMul(r: *Rig, a: std.mem.Allocator, im: c.Impl, o: u64, qg: u64, rows: usize) !Run {
    const n = rows * heads * hd;
    const out = try r.out(n * 2);
    const s = try r.out(n / 64 * 4);
    switch (im) {
        .native => try r.native.gateMul(r.s, o, qg, out.ptr, s.ptr, rows, heads),
        .triton => try r.replay.gateMul(o, qg, out.ptr, s.ptr, rows, heads, hd),
    }
    return .{ .y = try c.fetch(u16, r, a, out, n), .xs = try c.fetch(f32, r, a, s, n / 64) };
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
    var want: [hd]f64 = undefined;
    var lim: [hd]f64 = undefined;
    var flo: [hd]f64 = undefined;
    var sto: [hd]f64 = undefined;
    var sto_t: [hd]f64 = undefined;
    var sp: [2]usize = undefined;
    for (c.spans(rows, &sp)) |n| {
        const nat = try runGateMul(r, a, .native, ob.ptr, qb.ptr, n);
        const tr: ?Run = if (tri) try runGateMul(r, a, .triton, ob.ptr, qb.ptr, n) else null;
        for (0..n * heads) |vec| {
            for (&want, &lim, 0..) |*w, *l, e| {
                const gate = f(qg[(vec * 2 + 1) * hd + e]);
                const ov = f(o[vec * hd + e]);
                w.* = ov * aref.sigmoid(gate);
                flo[e] = @abs(ov) * q.flt_min;
                l.* = q.bound(w.*, c.sig(c.rel_sigmoid, gate), flo[e]);
            }
            const col = vec % heads * hd;
            for (&sto, &sto_t, 0..) |*s, *t, e| {
                s.* = f(nat.y[vec * hd + e]);
                t.* = if (tr) |x| f(x.y[vec * hd + e]) else 0;
                g.y.at(n).add(col + e, s.*, if (tr != null) t.* else null, want[e], lim[e], @max(@abs(want[e]), flo[e], c.floor));
            }
            const gs = hd / 64;
            const nat_row: c.Row = .{ .xs = nat.xs[vec * gs ..][0..gs], .y = &sto };
            const tri_row: ?c.Row = if (tr) |x| .{ .xs = x.xs[vec * gs ..][0..gs], .y = &sto_t } else null;
            c.xsRow(g, n, vec % heads * gs, nat_row, tri_row, &want, &lim);
        }
    }
}

/// The fp32 reduction bound of a random row's group: the deepest path's adds of u times the sum of the magnitudes.
fn depthBound(v: []const f64) f64 {
    return q.sumBound(v, c.depth_row);
}

/// group_sums of `rows` rows of `x` (stride `ld`, kdim values): fp32 sums against float64 sums; `dump` is Triton's.
fn sums(r: *Rig, o: *Out, x: []const u16, ld: usize, rows: usize, tri: bool, dump: ?[]const f32, bnd: *const fn ([]const f64) f64) !void {
    var arena = std.heap.ArenaAllocator.init(r.gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    defer r.release();
    const kg = kdim / 64;
    const xb = try r.up(base.bytes(x), true);
    const v = try a.alloc(f64, kdim);
    const want = try a.alloc(f64, kg);
    var sp: [2]usize = undefined;
    for (c.spans(rows, &sp)) |n| {
        const sn = try r.out(n * kg * 4);
        try r.native.groupSums(r.s, xb.ptr, sn.ptr, ld, n, kdim);
        const nat = try c.fetch(f32, r, a, sn, n * kg);
        var other: ?[]const f32 = if (dump) |d| d[0 .. n * kg] else null;
        if (dump == null and tri) {
            const stt = try r.out(n * kg * 4);
            try r.replay.groupSums(xb.ptr, stt.ptr, ld, n, kdim);
            other = try c.fetch(f32, r, a, stt, n * kg);
        }
        for (0..n) |row| {
            for (v, 0..) |*e, i| e.* = f(x[row * ld + i]);
            nref.groupSums(v, want);
            for (want, 0..) |w, j| {
                var scale: f64 = 0;
                for (v[64 * j ..][0..64]) |e| scale += @abs(e);
                const got_t: ?f64 = if (other) |t| t[row * kg + j] else null;
                o.at(n).add(j, nat[row * kg + j], got_t, w, bnd(v[64 * j ..][0..64]), @max(scale, c.floor));
            }
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
        const got_p = try c.fetch(i32, r, a, po, rows * glue.max_depth);
        const got_d = try c.fetch(i32, r, a, dp, rows);
        if (e.poke and k == 0) got_p[(rows - 1) * glue.max_depth] +%= 1;
        if (!std.mem.eql(i32, got_p, want_p) or !std.mem.eql(i32, got_d, want_d)) e.bad[k] += 1;
    }
}

/// Every kernel's outputs; the negative control moves one native column of each off its reference.
const Outs = struct {
    add0: Group,
    add1: Group,
    real_add0: Group,
    real_add1: Group,
    gated: Group,
    gate: Group,
    sums: Out,
    ladder: Out,
    real_sums: Out,
    embed: Out,
    paths: Exact = .{},

    fn init(a: std.mem.Allocator, mode: []const u8) !Outs {
        const norm = "halfUlp + 2^-19 |y|";
        const gate_text = "halfUlp + sigmoid slack |out| + flush";
        const sum_text = std.fmt.comptimePrint("{d}u sum|x|, u=2^-24", .{c.depth_row});
        return .{
            .add0 = try Group.init(a, mode, "add_rmsnorm[HAS_R=0]", "Y", norm, hidden, c.depth_row),
            .add1 = try Group.init(a, mode, "add_rmsnorm[HAS_R=1]", "Y", norm, hidden, c.depth_row),
            .real_add0 = try Group.init(a, mode, "add_rmsnorm[real h rows]", "Y", norm, hidden, c.depth_row),
            .real_add1 = try Group.init(a, mode, "add_rmsnorm[HAS_R=1, real rows]", "Y", norm, hidden, c.depth_row),
            .gated = try Group.init(a, mode, "gated_norm", "OUT", "halfUlp + norm and sigmoid slack |out| + flush", vh * dv, c.depth_head),
            .gate = try Group.init(a, mode, "gate_mul", "OUT", gate_text, heads * hd, c.depth_head),
            .sums = try Out.init(a, try std.fmt.allocPrint(a, "{s}:group_sums.XS", .{mode}), kdim / 64, sum_text),
            .ladder = try Out.init(a, try std.fmt.allocPrint(a, "{s}:group_sums[ladder rows].XS", .{mode}), kdim / 64, "sum gamma(d_i)|x_i|, d_i the adds on x_i's path"),
            .real_sums = try Out.init(a, try std.fmt.allocPrint(a, "{s}:group_sums[real y rows].XS", .{mode}), kdim / 64, sum_text),
            .embed = try Out.init(a, try std.fmt.allocPrint(a, "{s}:embed.OUT", .{mode}), kdim, "halfUlp + 2^-23 |out|"),
        };
    }

    fn groups(o: *Outs) [6]*Group {
        return .{ &o.add0, &o.add1, &o.real_add0, &o.real_add1, &o.gated, &o.gate };
    }

    fn perturb(o: *Outs, col: usize) void {
        for (o.groups()) |g| g.perturb(col);
        for ([_]*Out{ &o.sums, &o.ladder, &o.real_sums, &o.embed }) |x| x.p.perturb(col);
        o.paths.poke = true;
    }

    /// The first armed output that took part in the run and still passed.
    fn missed(o: *Outs) ?[]const u8 {
        for (o.groups()) |g| for (g.all()) |x| if (x.missed()) |name| return name;
        for ([_]*Out{ &o.sums, &o.ladder, &o.real_sums, &o.embed }) |x| if (x.missed()) |name| return name;
        return null;
    }

    fn report(o: *Outs) usize {
        var bad: usize = 0;
        for (o.groups()) |g| bad += g.report();
        for ([_]*Out{ &o.sums, &o.ladder, &o.real_sums, &o.embed }) |x| bad += x.report();
        if (o.paths.cases > 0) std.debug.print("F64 paths exact cases={d} native differ={d} triton differ={d}\n", .{ o.paths.cases, o.paths.bad[0], o.paths.bad[1] });
        return bad + o.paths.bad[0] + o.paths.bad[1];
    }
};

/// The dumps' rows: y rows through group_sums, h rows as x alone and (x, residual) pairs through add_rmsnorm.
const Dumps = struct {
    r: *Rig,
    o: *Outs,
    tri: bool,

    pub fn y(d: Dumps, rows: []const u16, xs: []const f32) !void {
        try sums(d.r, &d.o.real_sums, rows, kdim, rows.len / kdim, false, xs, &depthBound);
    }

    pub fn add(d: Dumps, rows: c.Real) !void {
        try addRms(d.r, if (rows.r != null) &d.o.real_add1 else &d.o.real_add0, 0, false, d.tri, rows);
    }
};

/// The served shapes on random rows: `rounds` of every row count, the kernels run in turn.
fn sweep(r: *Rig, o: *Outs, table: emb.Table, counts: []const usize, tri: bool, rounds: usize, stress: bool) !void {
    realistic_only = !stress;
    for (counts) |rows| for (0..@min(rounds, reps(rows))) |_| {
        try addRms(r, &o.add0, rows, false, tri, null);
        try addRms(r, &o.add1, rows, true, tri, null);
        try gated(r, &o.gated, rows, tri);
        try gateMul(r, &o.gate, rows, tri);
        if (!stress) try emb.embed(r, &o.embed, table, rows, tri);
    };
    for ([_]usize{ kdim, kdim + 16, 2 * kdim + 48 }) |ld| for (counts) |rows| for (0..@min(rounds, reps(rows))) |_| {
        if (rows > 128) continue;
        const x = try matrix(r, r.gpu.gpa, rows, ld, 1.0);
        defer r.gpu.gpa.free(x);
        try sums(r, &o.sums, x, ld, rows, tri, null, &depthBound);
    };
    if (!stress) for (counts) |rows| if (rows <= 128) {
        const x = try r.gpu.gpa.alloc(u16, rows * kdim);
        defer r.gpu.gpa.free(x);
        for (0..rows) |row| ladder.fill(x[row * kdim ..][0..kdim], row);
        try sums(r, &o.ladder, x, kdim, rows, tri, null, &ladder.bound);
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
    var table = try emb.Table.init(&r, a);
    defer table.deinit();
    std.debug.print("-- qwen-glue-f64: native vs float64, Triton {s}, oracle {s}\n", .{ if (tri) "replay" else "none", if (oracle != null) "dumps" else "none" });
    var o = try Outs.init(a, "realistic");
    try sweep(&r, &o, table, &row_counts, tri, 6, false);
    if (oracle) |dir| try real.walk(&r, dir, Dumps{ .r = &r, .o = &o, .tri = tri });
    var x = try Outs.init(a, "stress");
    try sweep(&r, &x, table, &row_counts, tri, 2, true);
    const bad = o.report() + x.report();
    // The negative control: one native column of every output 1.25 bounds off must fail every output.
    var ctl = try Outs.init(a, "control");
    ctl.perturb(control_col);
    try sweep(&r, &ctl, table, &.{3}, tri, 1, false);
    const missed = ctl.missed();
    if (missed) |name| std.debug.print("FAIL control: {s} accepted a perturbed native element\n", .{name});
    std.debug.print("   control: every output rejected its perturbed element, paths {s}\n", .{if (ctl.paths.bad[0] > 0) "rejected" else "accepted"});
    std.debug.print("RESULT qwen-glue-f64: {d} outside bounds or worse than Triton\n", .{bad});
    try check.expect(bad == 0 and missed == null and ctl.paths.bad[0] > 0, "qwen-glue-f64", .{});
}
