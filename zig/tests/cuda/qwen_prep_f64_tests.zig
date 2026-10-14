//! Qwen3.6's prep kernels (qwen_prep.cu: gdn_pre, attn_prep) against float64 references from the same input bytes.

const std = @import("std");
const cuda = @import("cuda");
const qwen27 = @import("qwen27");
const check = @import("check.zig");
const f64t = @import("qwen_f64.zig");
const base = @import("qwen_prep_tests.zig");

const tri = qwen27.triton;
const prep = qwen27.prep_native;
const Buf = base.Buf;
const Phased = f64t.Phased;
const keep = base.keep;
const C = base.gdn_shape.c;
const KH = base.gdn_shape.kh;
const VH = base.gdn_shape.vh;
const DK = base.gdn_shape.dk;
const H = base.attn_shape.heads;
const HKV = base.attn_shape.kv_heads;
const D = base.attn_shape.dim;
const HALF = base.attn_shape.half;
const rows_list = [_]usize{ 1, 5, 64, 513, 2048 };
// a prompt-sized call is also run on its first rows of this many, the decode bucket
const decode_rows = f64t.decode_rows;
const serve_pos_max: u64 = 262143; // the longest context the model serves; rows past it are the stress class
const eps24 = 0x1p-24;

const Group = struct {
    pairs: []Phased,

    fn init(gpa: std.mem.Allocator, prefix: []const u8, names: []const []const u8, cols: []const usize) !Group {
        const pairs = try gpa.alloc(Phased, names.len);
        for (pairs, names, cols) |*p, nm, c| {
            const full = try std.fmt.allocPrint(gpa, "{s}{s}", .{ prefix, nm });
            defer gpa.free(full);
            p.* = try Phased.init(gpa, full, c);
        }
        return .{ .pairs = pairs };
    }
};

const Rig = struct {
    gpu: check.Gpu,
    s: cuda.Stream,
    native: *const prep.Native,
    replay: ?tri.Tri,
    rows_done: usize = 0,

    fn alloc(r: *Rig, n: usize, shift: bool) !Buf {
        const b = try cuda.DeviceBuffer.alloc(r.gpu.d, n + 64);
        try b.fill8(0xff, r.s.handle);
        return .{ .b = b, .off = if (shift) 36 else 32 };
    }

    fn up(r: *Rig, bytes: []const u8, shift: bool) !Buf {
        const x = try r.alloc(bytes.len, shift);
        try r.s.synchronize();
        try x.b.upload(x.off, bytes);
        return x;
    }

    fn fetch(r: *Rig, b: Buf, n: usize) ![]u8 {
        try r.s.synchronize();
        const all = try check.download(r.gpu, b.b);
        defer r.gpu.gpa.free(all);
        return r.gpu.gpa.dupe(u8, all[b.off..][0..n]);
    }
};

const Align = struct {
    multi: bool = false,
    sh: [16]bool = @splat(false),

    fn of(v: anytype, names: []const []const u8) Align {
        var a: Align = .{};
        for (names, 0..) |n, i| a.sh[i] = base.shifted(v, n);
        return a;
    }
};

const gdn_names = [_][]const u8{ "QKV", "CS", "CW", "WIN", "A", "B", "ALOG", "DTB", "SID", "Q", "K", "V", "G", "BETA" };
const attn_names = [_][]const u8{ "QG", "KV", "QN", "KN", "POS", "INV", "QOUT", "KOUT" };

// ---- gdn_pre: conv over [state; qkv], SiLU, q/k RMS norm, v, and g and beta per value head

const GdnRow = struct {
    q: [KH * DK]f64,
    qb: [KH * DK]f64,
    k: [KH * DK]f64,
    kb: [KH * DK]f64,
    v: [VH * DK]f64,
    vb: [VH * DK]f64,
    g: [VH]f64,
    gb: [VH]f64,
    beta: [VH]f64,
    bb: [VH]f64,
    // relative error is over max(|want|, 2^-7 of the head vector's largest |want|): one bf16 ulp of the vector
    qf: [KH]f64,
    kf: [KH]f64,
    vf: [VH]f64,
    c: [C]f64,
    cond: [C]f64,
};

// The float64 row: bounds are bf16 spacing at the value (rounding points counted below) plus the fp32 condition.
fn gdnRow(d: *const base.Gdn, row: usize, kind: usize, multi: bool, o: *GdnRow) void {
    const cs_row: usize = if (multi) @as(usize, @intCast(d.sid[kind][row])) * keep else 0;
    for (0..C) |ch| {
        var sum: f64 = 0;
        var mag: f64 = 0;
        for (0..keep + 1) |j| {
            const src: usize = @intCast(d.win[kind][row * (keep + 1) + j]);
            const x = if (src < keep) f64t.bf(d.cs[(cs_row + src) * C + ch]) else f64t.bf(d.qkv[(src - keep) * C + ch]);
            const t = x * f64t.bf(d.cw[ch * (keep + 1) + j]);
            sum += t;
            mag += @abs(t);
        }
        o.c[ch] = sum / (1 + @exp(-sum));
        o.cond[ch] = 1.1 * 4 * eps24 * mag; // four fma roundings of the window sum, through SiLU's slope (at most 1.1)
    }
    for (0..2 * KH) |head| {
        const cs = o.c[head * DK ..][0..DK];
        var ss: f64 = 0;
        var cmax: f64 = 0;
        for (cs, o.cond[head * DK ..][0..DK]) |x, cd| {
            ss += x * x;
            cmax = @max(cmax, cd);
        }
        const scale: f64 = if (head < KH) 1.0 / 128.0 else 1.0 / @sqrt(128.0);
        const rs = scale / @sqrt(ss / DK + 1e-6);
        const want = if (head < KH) o.q[head * DK ..][0..DK] else o.k[(head - KH) * DK ..][0..DK];
        const bound = if (head < KH) o.qb[head * DK ..][0..DK] else o.kb[(head - KH) * DK ..][0..DK];
        // three bf16 ulps: the bf16 SiLU feeding the norm, its coupling through the mean square, the bf16 store
        var big: f64 = 0;
        for (cs, want, bound) |x, *w, *b| {
            w.* = x * rs;
            b.* = 3 * f64t.ulpBf16(w.*) + 0x1p-21 * @abs(w.*) + 2 * rs * cmax;
            big = @max(big, @abs(w.*));
        }
        if (head < KH) o.qf[head] = big * 0x1p-7 else o.kf[head - KH] = big * 0x1p-7;
    }
    for (0..VH) |hv| {
        o.vf[hv] = 0;
        for (0..DK) |i| {
            const x = o.c[(2 * KH + hv) * DK + i];
            o.vf[hv] = @max(o.vf[hv], @abs(x) * 0x1p-7);
            o.v[hv * DK + i] = x;
            o.vb[hv * DK + i] = f64t.ulpBf16(x) + 0x1p-21 * @abs(x) + o.cond[(2 * KH + hv) * DK + i];
        }
        const a = f64t.bf(d.a[row * VH + hv]) + @as(f32, d.dtb[hv]);
        const sp = if (a > 0) a + std.math.log1p(@exp(-a)) else std.math.log1p(@exp(a));
        const ea = @exp(@as(f64, d.alog[hv]));
        const x = sp * ea;
        o.g[hv] = @exp(-x);
        // fp32 intrinsics: ex2 and log polynomial at 2^-22, carried through the exponent by x; 1 + exp(a) absorbs 2^-24
        o.gb[hv] = o.g[hv] * (0x1p-20 * (1 + x) + ea * eps24 * (1 + @abs(a))) + f64t.flt_min;
        const b = f64t.bf(d.b[row * VH + hv]);
        o.beta[hv] = 1 / (1 + @exp(-b));
        o.bb[hv] = o.beta[hv] * (0x1p-20 + @abs(b) * 0x1p-22) + f64t.flt_min;
    }
}

const GdnIn = struct { qkv: u64, cs: u64, cw: u64, win: u64, sid: ?u64, a: u64, b: u64, alog: u64, dtb: u64 };
const GdnBytes = [2][5][]const u8; // [native, replay][q, k, v, g, beta]

fn gdnSizes(rows: usize) [5]usize {
    return .{ rows * KH * DK * 2, rows * KH * DK * 2, rows * VH * DK * 2, rows * VH * 4, rows * VH * 4 };
}

fn gdnLaunch(r: *Rig, i: GdnIn, outs: *const [2][5]Buf, rows: usize) !void {
    const n = outs[0];
    try r.native.gdnPre(r.s, i.qkv, i.cs, i.cw, i.win, i.sid, i.a, i.b, i.alog, i.dtb, n[0].ptr(), n[1].ptr(), n[2].ptr(), n[3].ptr(), n[4].ptr(), rows, KH, VH, DK, C, keep);
    const o = outs[1];
    if (r.replay) |t| try t.gdnPre(i.qkv, i.cs, i.cw, i.win, i.sid, i.a, i.b, i.alog, i.dtb, o[0].ptr(), o[1].ptr(), o[2].ptr(), o[3].ptr(), o[4].ptr(), rows, base.gdn_shape);
}

fn gdnFetch(r: *Rig, outs: *const [2][5]Buf, rows: usize, bytes: *GdnBytes) !void {
    const sizes = gdnSizes(rows);
    for (bytes, outs, 0..) |*bs, *os, w| for (bs, os, sizes) |*x, o, n| {
        x.* = if (w == 1 and r.replay == null) &[_]u8{} else try r.fetch(o, n);
    };
}

fn gdnFree(gpa: std.mem.Allocator, bytes: *const GdnBytes) void {
    for (bytes) |bs| for (bs) |x| if (x.len > 0) gpa.free(x);
}

// Row `rw` of a launch of `rows` rows (which picks the phase) against the reference row.
fn gdnAdd(grp: *Group, rows: usize, bytes: *const GdnBytes, tri_on: bool, row: *const GdnRow, rw: usize) void {
    const pr = grp.pairs;
    for (0..KH * DK) |i| {
        const at = rw * KH * DK + i;
        pr[0].at(rows).add(i % DK, f64t.bfAt(bytes[0][0], at), if (tri_on) f64t.bfAt(bytes[1][0], at) else null, row.q[i], row.qb[i], row.qf[i / DK]);
        pr[1].at(rows).add(i % DK, f64t.bfAt(bytes[0][1], at), if (tri_on) f64t.bfAt(bytes[1][1], at) else null, row.k[i], row.kb[i], row.kf[i / DK]);
    }
    for (0..VH * DK) |i| {
        const at = rw * VH * DK + i;
        pr[2].at(rows).add(i % DK, f64t.bfAt(bytes[0][2], at), if (tri_on) f64t.bfAt(bytes[1][2], at) else null, row.v[i], row.vb[i], row.vf[i / DK]);
    }
    for (0..VH) |hv| {
        const at = rw * VH + hv;
        pr[3].at(rows).add(hv, f64t.f32At(bytes[0][3], at), if (tri_on) f64t.f32At(bytes[1][3], at) else null, row.g[hv], row.gb[hv], 1e-30);
        pr[4].at(rows).add(hv, f64t.f32At(bytes[0][4], at), if (tri_on) f64t.f32At(bytes[1][4], at) else null, row.beta[hv], row.bb[hv], 1e-30);
    }
}

fn gdnPass(r: *Rig, d: *const base.Gdn, al: Align, grp: *Group, row: *GdnRow) !void {
    const gpa = r.gpu.gpa;
    var qkv = try r.up(std.mem.sliceAsBytes(d.qkv), al.sh[0]);
    defer qkv.free();
    var cs = try r.up(std.mem.sliceAsBytes(if (al.multi) d.cs else d.cs[0 .. keep * C]), al.sh[1]);
    defer cs.free();
    var cw = try r.up(std.mem.sliceAsBytes(d.cw), al.sh[2]);
    defer cw.free();
    var a = try r.up(std.mem.sliceAsBytes(d.a), al.sh[4]);
    defer a.free();
    var b = try r.up(std.mem.sliceAsBytes(d.b), al.sh[5]);
    defer b.free();
    var alog = try r.up(std.mem.sliceAsBytes(&d.alog), al.sh[6]);
    defer alog.free();
    var dtb = try r.up(std.mem.sliceAsBytes(&d.dtb), al.sh[7]);
    defer dtb.free();
    for (0..2) |kind| {
        var win = try r.up(std.mem.sliceAsBytes(d.win[kind]), al.sh[3]);
        defer win.free();
        var sid = try r.up(std.mem.sliceAsBytes(d.sid[kind]), al.sh[8]);
        defer sid.free();
        const gin: GdnIn = .{ .qkv = qkv.ptr(), .cs = cs.ptr(), .cw = cw.ptr(), .win = win.ptr(), .sid = if (al.multi) sid.ptr() else null, .a = a.ptr(), .b = b.ptr(), .alog = alog.ptr(), .dtb = dtb.ptr() };
        for (rows_list) |rows| {
            var outs: [2][5]Buf = undefined;
            for (&outs) |*os| for (os, gdnSizes(rows), 0..) |*x, n, i| {
                x.* = try r.alloc(n, al.sh[9 + i]);
            };
            defer for (&outs) |*os| for (os) |*x| x.free();
            try gdnLaunch(r, gin, &outs, rows);
            var bytes: GdnBytes = undefined;
            try gdnFetch(r, &outs, rows, &bytes);
            defer gdnFree(gpa, &bytes);
            for (0..rows) |rw| {
                gdnRow(d, rw, kind, al.multi, row);
                gdnAdd(grp, rows, &bytes, r.replay != null, row, rw);
                r.rows_done += 1;
            }
            if (rows <= decode_rows) continue;
            // the same rows through the decode-sized launch: the first rows of the same buffers, rows = decode_rows
            try gdnLaunch(r, gin, &outs, decode_rows);
            var head: GdnBytes = undefined;
            try gdnFetch(r, &outs, decode_rows, &head);
            defer gdnFree(gpa, &head);
            for (0..decode_rows) |rw| {
                gdnRow(d, rw, kind, al.multi, row);
                gdnAdd(grp, decode_rows, &head, r.replay != null, row, rw);
            }
        }
    }
}

// ---- attn_prep: q and k RMS norm, then rotate-half rotary at the row's position

const AttnRow = struct { want: [D]f64, bound: [D]f64, xn: [D]f64, floor: f64 };

fn attnHead(t: *const base.Attn, row: usize, head: usize, o: *AttnRow) void {
    const src = if (head < H) t.qg[(row * H + head) * 2 * D ..][0..D] else t.kv[(row * HKV + head - H) * D ..][0..D];
    const wt = if (head < H) &t.qn else &t.kn;
    var ss: f64 = 0;
    for (src) |x| ss += f64t.bf(x) * f64t.bf(x);
    const inv = 1 / @sqrt(ss / D + 1e-6);
    for (src, wt, &o.xn) |x, w, *n| n.* = f64t.bf(x) * inv * f64t.bf(w);
    const pos: f64 = @floatFromInt(t.pos[row]);
    for (0..D) |i| {
        if (i >= 2 * HALF) {
            o.want[i] = o.xn[i];
            o.bound[i] = f64t.ulpBf16(o.xn[i]) + 0x1p-21 * @abs(o.xn[i]);
            continue;
        }
        const j = i % HALF;
        const p = if (i < HALF) i + HALF else i - HALF;
        const ang = pos * @as(f64, t.inv[j]);
        const cs = @cos(ang);
        const sn = @sin(ang);
        o.want[i] = if (i < HALF) o.xn[i] * cs - o.xn[p] * sn else o.xn[i] * cs + o.xn[p] * sn;
        // the pair's magnitude: the scale a cancelling rotation keeps
        const m = @sqrt(o.xn[i] * o.xn[i] + o.xn[p] * o.xn[p]);
        // two bf16 ulps of it (norm output, store), cosf and sinf at 2^-21, the fp32 angle (int to float, product)
        o.bound[i] = 2 * f64t.ulpBf16(m) + m * (0x1p-21 + 0x1p-23 * @abs(ang));
    }
    var big: f64 = 0;
    for (o.want) |x| big = @max(big, @abs(x));
    o.floor = big * 0x1p-7;
}

const AttnBytes = [2][2][]const u8; // [native, replay][q, k]

fn attnSizes(rows: usize) [2]usize {
    return .{ rows * H * D * 2, rows * HKV * D * 2 };
}

fn attnLaunch(r: *Rig, i: [6]u64, outs: *const [2][2]Buf, rows: usize) !void {
    try r.native.attnPrep(r.s, i[0], i[1], i[2], i[3], i[4], i[5], outs[0][0].ptr(), outs[0][1].ptr(), rows, H, HKV, D, HALF, 1e-6);
    if (r.replay) |tr| try tr.attnPrep(i[0], i[1], i[2], i[3], i[4], i[5], outs[1][0].ptr(), outs[1][1].ptr(), rows, base.attn_shape, 1e-6);
}

fn attnFetch(r: *Rig, outs: *const [2][2]Buf, rows: usize, bytes: *AttnBytes) !void {
    const sizes = attnSizes(rows);
    for (bytes, outs, 0..) |*bs, *os, w| for (bs, os, sizes) |*x, o, n| {
        x.* = if (w == 1 and r.replay == null) &[_]u8{} else try r.fetch(o, n);
    };
}

fn attnFree(gpa: std.mem.Allocator, bytes: *const AttnBytes) void {
    for (bytes) |bs| for (bs) |x| if (x.len > 0) gpa.free(x);
}

// The first `rows` rows of a launch of `rows` rows (which picks the phase) against the reference.
fn attnAdd(grp: *Group, t: *const base.Attn, rows: usize, bytes: *const AttnBytes, tri_on: bool, row: *AttnRow) void {
    for (0..rows) |rw| {
        const stress = @abs(@as(i64, t.pos[rw])) > serve_pos_max;
        for (0..H + HKV) |head| {
            attnHead(t, rw, head, row);
            const which: usize = (if (head < H) @as(usize, 0) else 1) + (if (stress) @as(usize, 2) else 0);
            const out: usize = if (head < H) (rw * H + head) * D else (rw * HKV + head - H) * D;
            const k: usize = if (head < H) 0 else 1;
            for (0..D) |i| {
                const tg: ?f64 = if (tri_on) f64t.bfAt(bytes[1][k], out + i) else null;
                grp.pairs[which].at(rows).add(i, f64t.bfAt(bytes[0][k], out + i), tg, row.want[i], row.bound[i], row.floor);
            }
        }
    }
}

fn attnPass(r: *Rig, t: *const base.Attn, al: Align, grp: *Group, row: *AttnRow) !void {
    const gpa = r.gpu.gpa;
    var qg = try r.up(std.mem.sliceAsBytes(t.qg), al.sh[0]);
    defer qg.free();
    var kv = try r.up(std.mem.sliceAsBytes(t.kv), al.sh[1]);
    defer kv.free();
    var qn = try r.up(std.mem.sliceAsBytes(&t.qn), al.sh[2]);
    defer qn.free();
    var kn = try r.up(std.mem.sliceAsBytes(&t.kn), al.sh[3]);
    defer kn.free();
    var pos = try r.up(std.mem.sliceAsBytes(t.pos), al.sh[4]);
    defer pos.free();
    var inv = try r.up(std.mem.sliceAsBytes(&t.inv), al.sh[5]);
    defer inv.free();
    const ain = [6]u64{ qg.ptr(), kv.ptr(), qn.ptr(), kn.ptr(), pos.ptr(), inv.ptr() };
    for (rows_list) |rows| {
        var outs: [2][2]Buf = undefined;
        for (&outs) |*os| for (os, attnSizes(rows), 0..) |*x, n, i| {
            x.* = try r.alloc(n, al.sh[6 + i]);
        };
        defer for (&outs) |*os| for (os) |*x| x.free();
        try attnLaunch(r, ain, &outs, rows);
        var bytes: AttnBytes = undefined;
        try attnFetch(r, &outs, rows, &bytes);
        defer attnFree(gpa, &bytes);
        attnAdd(grp, t, rows, &bytes, r.replay != null, row);
        r.rows_done += rows;
        if (rows <= decode_rows) continue;
        // the same rows through the decode-sized launch: the first rows of the same buffers, rows = decode_rows
        try attnLaunch(r, ain, &outs, decode_rows);
        var head: AttnBytes = undefined;
        try attnFetch(r, &outs, decode_rows, &head);
        defer attnFree(gpa, &head);
        attnAdd(grp, t, decode_rows, &head, r.replay != null, row);
    }
}

/// Realistic positions: the serving range (up to 262143) and the generator's decode runs, a few stress rows past it.
fn servePositions(t: *base.Attn, rng: std.Random) void {
    for (t.pos, 0..) |*p, i| {
        if (@abs(@as(i64, p.*)) > serve_pos_max and i % 3 != 0) p.* = @intCast(rng.uintLessThan(u32, @intCast(serve_pos_max + 1)));
        if (i % 97 == 5) p.* = @intCast(serve_pos_max - i % 7);
    }
}

fn realisticAlog(g: *base.Gdn, rng: std.Random) void {
    for (&g.alog) |*al| al.* = @log(1 + rng.float(f32) * 15);
}

fn heavyAttn(t: *base.Attn, rng: std.Random) void {
    const sc = [_]f32{ 0.01, 1, 30, 1000 };
    base.fillScaled(rng, t.qg, 256, &sc);
    base.fillScaled(rng, t.kv, 256, &sc);
}

fn report(failed: *bool, grp: *Group, strict: bool, text: []const u8) !void {
    for (grp.pairs) |*p| try f64t.gate(failed, p.finish(text, strict));
}

pub fn run(gpu: check.Gpu, aot_dir: ?[]const u8) !void {
    const gpa = gpu.gpa;
    var set: ?cuda.aot.Set = if (aot_dir) |dir| try cuda.aot.Set.load(gpa, gpu.io, gpu.d, gpu.ctx.device, dir) else null;
    defer if (set) |*s| s.deinit();
    const s = try cuda.Stream.init(gpu.d, true);
    const control: ?[]u8 = if (std.c.getenv("TF_QWEN_PREP_FATBIN")) |p| try std.Io.Dir.cwd().readFileAlloc(gpu.io, std.mem.span(p), gpa, .limited(1 << 26)) else null;
    defer if (control) |c| gpa.free(c);
    var native = if (control) |c| try prep.Native.loadImage(gpu.d, try gpu.ctx.capability(), c) else try prep.Native.load(gpu.d, try gpu.ctx.capability());
    defer native.deinit();
    var r: Rig = .{ .gpu = gpu, .s = s, .native = &native, .replay = if (set) |*x| .{ .set = x, .s = s } else null };
    const gdn_cols = [_]usize{ DK, DK, DK, VH, VH };
    const gdn_out = [_][]const u8{ "q", "k", "v", "g", "beta" };
    const attn_cols = [_]usize{ D, D, D, D };
    const attn_out = [_][]const u8{ "q", "k", "q_stress", "k_stress" };
    var gdn_g = try Group.init(gpa, "gdn_pre.", &gdn_out, &gdn_cols);
    var gdn_w = try Group.init(gpa, "gdn_pre.wild.", &gdn_out, &gdn_cols);
    var attn_g = try Group.init(gpa, "attn_prep.", &attn_out, &attn_cols);
    var attn_w = try Group.init(gpa, "attn_prep.wild.", &attn_out, &attn_cols);
    const grow = try gpa.create(GdnRow);
    defer gpa.destroy(grow);
    const arow = try gpa.create(AttnRow);
    defer gpa.destroy(arow);
    var prng = std.Random.DefaultPrng.init(0xf64_0a11);
    const rng = prng.random();
    // data sets: 0 serving magnitudes, 1 heavy tails (bound asserted), 2 the byte-equality extremes (Triton only)
    for (0..3) |ds| {
        if (ds == 2 and set == null) break;
        var g = try base.Gdn.make(gpa, rng, ds != 0);
        defer g.free(gpa);
        var t = try base.Attn.make(gpa, rng, ds == 2);
        defer t.free(gpa);
        if (ds != 2) {
            realisticAlog(&g, rng);
            servePositions(&t, rng);
        }
        if (ds == 1) heavyAttn(&t, rng);
        const gg = if (ds == 2) &gdn_w else &gdn_g;
        const ag = if (ds == 2) &attn_w else &attn_g;
        if (set) |*st| {
            for (st.variants) |*v| {
                if (std.mem.eql(u8, v.spec.@"fn", "_gdn_pre")) {
                    var al = Align.of(v.*, &gdn_names);
                    al.multi = base.constOf(v.*, "MULTI") == 1;
                    try gdnPass(&r, &g, al, gg, grow);
                } else if (std.mem.eql(u8, v.spec.@"fn", "_attn_prep")) {
                    try attnPass(&r, &t, Align.of(v.*, &attn_names), ag, arow);
                }
            }
        } else {
            for ([_]bool{ false, true }) |multi| try gdnPass(&r, &g, .{ .multi = multi }, gg, grow);
            try attnPass(&r, &t, .{}, ag, arow);
        }
        std.debug.print("data set {d}: {d} rows compared so far\n", .{ ds, r.rows_done });
    }
    const tri_on = set != null;
    var failed = false;
    try report(&failed, &gdn_g, true, "3 bf16 ulp (silu round, norm coupling, store) + 2^-21 + conv fp32 chain; v 1 ulp; g,beta fp32 intrinsics");
    try report(&failed, &attn_g, true, "2 bf16 ulp of the rotated pair (1 plain) + cosf/sinf 2^-21 + fp32 angle 2^-23*|pos*inv|");
    if (tri_on) {
        try report(&failed, &gdn_w, false, "extremes: native no worse than Triton, not bounded");
        try report(&failed, &attn_w, false, "extremes: native no worse than Triton, not bounded");
    }
    try check.expect(!failed, "qwen-prep-f64: an output left its float64 bound or is worse than Triton", .{});
    check.pass("qwen-prep-f64: gdn_pre and attn_prep within their float64 bounds over {d} rows (Triton compared: {})", .{ r.rows_done, tri_on });
}
