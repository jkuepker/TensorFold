//! Qwen3.6 tree attention (_shared, _tail, _merge) against float64; the mask comes from the parents, not the plan.

const std = @import("std");
const cuda = @import("cuda");
const qwen27 = @import("qwen27");
const check = @import("check.zig");
const f64t = @import("qwen_f64.zig");
const att = @import("qwen_attention_tests.zig");
const ref = @import("qwen_attention_f64_ref.zig");

const tri = qwen27.triton;
const H = att.H;
const D = att.D;
const HK = att.HK;
const grid_p = [_]usize{ 0, 1, 511, 512, 513, 2047, 2048, 4096, 6144, 8191 };
const grid_w = [_]usize{ 1, 4, 17, 64 };
const wide_p = [_]usize{ 0, 511, 4096, 8191 }; // the widest window (128 rows) is dear for the float64 reference: a few lengths

const Ctx = struct {
    rig: *att.Rig,
    gpa: std.mem.Allocator,
    out: f64t.Pair, // the attention output, by head dim
    po: f64t.Pair, // each partial slot's o / l, by head dim
    lse: f64t.Pair, // each partial slot's m + ln l, by query head
    scores: []f64,
    row: *ref.Row,
    cases: usize = 0,
    skipped: usize = 0,
    rows: usize = 0,
    slots: usize = 0,
    bad_empty: usize = 0,
    triton_cases: usize = 0,
};

fn fetch(c: *Ctx, ptr: u64, bytes: usize) ![]u8 {
    const out = try c.gpa.alloc(u8, bytes);
    errdefer c.gpa.free(out);
    if (bytes > 0) try (cuda.DeviceBuffer{ .d = c.rig.gpu.d, .ptr = ptr, .len = bytes }).download(0, out);
    return out;
}

const Raw = struct { out: []u8, po: []u8, pm: []u8, pl: []u8 };

fn fetchResult(c: *Ctx, out: cuda.DeviceBuffer, part: att.Buffers, rows: usize, slots: usize) !Raw {
    const unit = slots * rows * H;
    return .{ .out = try fetch(c, out.ptr, rows * H * D * 2), .po = try fetch(c, part.o.ptr, unit * D * 4), .pm = try fetch(c, part.m.ptr, unit * 4), .pl = try fetch(c, part.l.ptr, unit * 4) };
}

fn freeRaw(c: *Ctx, r: Raw) void {
    for ([_][]u8{ r.out, r.po, r.pm, r.pl }) |x| c.gpa.free(x);
}

fn partialAt(r: Raw, idx: usize) struct { lse: f64, pl: f64, pm: f64 } {
    const pm = f64t.f32At(r.pm, idx);
    const pl = f64t.f32At(r.pl, idx);
    return .{ .lse = pm + @log(pl), .pl = pl, .pm = pm };
}

// One stream's rows against the reference: the output, then every slot the row's keys land in.
fn compare(c: *Ctx, s: *const ref.Stream, q: []const u8, nat: Raw, rep: ?Raw, rows: usize) !void {
    const width = rows;
    const nslots = tri.slots(s.p, s.w);
    for (0..s.w) |j| {
        const node = s.start + j;
        for (0..H) |head| {
            ref.row(s, j, head, q[(node * H + head) * D * 2 ..], c.scores, c.row);
            const r = c.row;
            for (0..D) |d| {
                const at = (node * H + head) * D + d;
                const tg: ?f64 = if (rep) |x| f64t.bfAt(x.out, at) else null;
                c.out.add(d, f64t.bfAt(nat.out, at), tg, r.out[d], r.out_bound[d], r.floor[d]);
            }
            for (0..r.slots) |slot| {
                if (!r.live[slot]) continue;
                const idx = (slot * width + node) * H + head;
                const pn = partialAt(nat, idx);
                const pt: ?@TypeOf(pn) = if (rep) |x| partialAt(x, idx) else null;
                c.lse.add(head, pn.lse, if (pt) |t| t.lse else null, r.lse[slot], r.lse_bound, 1);
                for (0..D) |d| {
                    const got = f64t.f32At(nat.po, idx * D + d) / pn.pl;
                    const tg: ?f64 = if (rep) |x| f64t.f32At(x.po, idx * D + d) / pt.?.pl else null;
                    c.po.add(d, got, tg, r.onorm[slot][d], r.onorm_bound[d], r.floor[d]);
                }
            }
            for (r.slots..nslots) |slot| { // chunks past this row's last key: written empty (l = 0, m = -inf)
                const idx = (slot * width + node) * H + head;
                for ([_]?Raw{ nat, rep }) |x| if (x) |y| {
                    const p = partialAt(y, idx);
                    if (!(p.pl == 0 and p.pm == -std.math.inf(f64))) c.bad_empty += 1;
                };
            }
            c.rows += 1;
        }
    }
}

fn runCase(c: *Ctx, specs: []const att.Spec, pool: usize) !void {
    const rig = c.rig;
    var par: [16][128]i32 = undefined;
    try rig.load(specs, pool);
    try att.inputs(rig, pool == 1);
    const sg = try att.stage(rig, specs, &par);
    for (rig.part) |b| try rig.clear(b);
    for (rig.out) |b| try b.fill8(0xff, rig.s.handle);
    const offs = sg.r.offsets(3, 0);
    var have_tri = false;
    if (rig.k.triton != null) {
        const t = rig.replay();
        if (t.attention(rig.q.ptr, rig.kn.ptr, rig.vn.ptr, rig.scratch.origin, offs, sg.r.attn, rig.part[0].parts(), rig.out[0].ptr, att.attn)) |_| {
            have_tri = true;
        } else |e| if (e != error.MissingTritonVariant) return e else c.skipped += 1;
    }
    const tn: tri.Tri = .{ .set = if (rig.k.triton) |*x| x else null, .s = rig.s, .tree = rig.nat };
    try tn.attention(rig.q.ptr, rig.kn.ptr, rig.vn.ptr, rig.scratch.origin, offs, sg.r.attn, rig.part[1].parts(), rig.out[1].ptr, att.attn);
    try rig.s.synchronize();
    const gpa = c.gpa;
    const q = try fetch(c, rig.q.ptr, sg.rows * H * D * 2);
    defer gpa.free(q);
    const kn = try fetch(c, rig.kn.ptr, sg.rows * HK * D * 2);
    defer gpa.free(kn);
    const vn = try fetch(c, rig.vn.ptr, sg.rows * HK * D * 2);
    defer gpa.free(vn);
    const nat = try fetchResult(c, rig.out[1], rig.part[1], sg.rows, sg.slots);
    defer freeRaw(c, nat);
    const rep: ?Raw = if (have_tri) try fetchResult(c, rig.out[0], rig.part[0], sg.rows, sg.slots) else null;
    defer if (rep) |x| freeRaw(c, x);
    var start: usize = 0;
    for (specs, 0..) |sp, i| {
        const seq = &rig.seqs[i];
        const kc = try fetch(c, seq.keys[3], sp.p * ref.kv_row * 2);
        defer gpa.free(kc);
        const vc = try fetch(c, seq.values[3], sp.p * ref.kv_row * 2);
        defer gpa.free(vc);
        var s = try ref.Stream.build(gpa, sp.p, sp.w, start, tri.groups(sp.p, sp.w), par[i][0..sp.w], kc, vc, kn, vn);
        defer s.deinit(gpa);
        try compare(c, &s, q, nat, rep, sg.rows);
        start += sp.w;
    }
    c.cases += 1;
    if (have_tri) c.triton_cases += 1;
    c.slots += sg.slots;
}

fn guarded(c: *Ctx, specs: []const att.Spec, pool: usize) !void {
    runCase(c, specs, pool) catch |e| {
        var lb: [160]u8 = undefined;
        std.debug.print("  case failed: {t} ({s})\n", .{ e, att.label(&lb, specs, pool) });
        return e;
    };
}

fn grid(c: *Ctx) !void {
    var n: usize = 0;
    for (grid_p) |p| for (grid_w) |w| {
        const shift: u8 = @intCast(n % 5);
        try guarded(c, &.{.{ .p = p, .w = w, .k = shift, .v = (shift * 3) % 5 }}, n % 4 / 3);
        try guarded(c, &.{.{ .p = p, .w = w, .kind = .bushy, .k = (shift + 2) % 5, .v = shift }}, 0);
        n += 1;
    };
    for (wide_p) |p| {
        try guarded(c, &.{.{ .p = p, .w = 128, .k = @intCast(n % 5) }}, 0);
        try guarded(c, &.{.{ .p = p, .w = 128, .kind = .bushy, .v = @intCast(n % 5) }}, n % 2);
        n += 1;
    }
}

fn lanes(c: *Ctx, count: usize) !void {
    const rng = c.rig.prng.random();
    var specs: [16]att.Spec = undefined;
    for (0..count) |_| {
        const s = 2 + rng.uintLessThan(usize, 5);
        for (specs[0..s]) |*sp| sp.* = .{ .p = att.committed[rng.uintLessThan(usize, att.committed.len)], .w = att.lane_widths[rng.uintLessThan(usize, att.lane_widths.len)], .kind = if (rng.boolean()) .chain else .bushy, .k = rng.uintLessThan(u8, 5), .v = rng.uintLessThan(u8, 5) };
        try guarded(c, specs[0..s], rng.uintLessThan(usize, 2));
    }
    for (0..2) |_| { // every stream the engine holds
        for (&specs) |*sp| sp.* = .{ .p = att.committed[rng.uintLessThan(usize, att.committed.len)], .w = att.lane_widths[rng.uintLessThan(usize, att.lane_widths.len)], .kind = if (rng.boolean()) .chain else .bushy, .k = rng.uintLessThan(u8, 5), .v = rng.uintLessThan(u8, 5) };
        try guarded(c, &specs, rng.uintLessThan(usize, 2));
    }
}

pub fn run(gpu: check.Gpu, aot_dir: ?[]const u8, rounds: usize) !void {
    var env: att.Env = undefined;
    try env.init(gpu, aot_dir);
    defer env.deinit();
    const gpa = gpu.gpa;
    const row = try gpa.create(ref.Row);
    defer gpa.destroy(row);
    var c: Ctx = .{
        .rig = &env.rig,
        .gpa = gpa,
        .out = try f64t.Pair.init(gpa, "tree_attention.out", D),
        .po = try f64t.Pair.init(gpa, "tree_attention.partial_o_over_l", D),
        .lse = try f64t.Pair.init(gpa, "tree_attention.partial_lse", H),
        .scores = try gpa.alloc(f64, 8192 + 256),
        .row = row,
    };
    defer gpa.free(c.scores);
    try grid(&c);
    std.debug.print("grid: {d} cases, {d} rows x heads compared so far\n", .{ c.cases, c.rows });
    try lanes(&c, rounds);
    std.debug.print("lanes: {d} cases, {d} with Triton, {d} rows x heads\n", .{ c.cases, c.triton_cases, c.rows });
    var failed = false;
    try f64t.gate(&failed, c.out.finish("3 bf16 half-ulps of max|v| (rounded p, unrounded l, bf16 store) + 2x the fp32 score error", true));
    try f64t.gate(&failed, c.po.finish("2 bf16 half-ulps of max|v| (rounded p, unrounded l) + 2x the fp32 score error", true));
    try f64t.gate(&failed, c.lse.finish("the fp32 score error + 2^-20 for the fp32 exp sum", true));
    try check.expect(!failed, "qwen-attention-f64: an output left its float64 bound or is worse than Triton", .{});
    try check.expect(c.bad_empty == 0, "{d} empty partial slots hold a key", .{c.bad_empty});
    check.pass("qwen-attention-f64: tree attention within its float64 bounds over {d} cases, {d} rows x heads (Triton compared in {d}, {d} skipped)", .{ c.cases, c.rows, c.triton_cases, c.skipped });
}
