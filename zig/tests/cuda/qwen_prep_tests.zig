//! Qwen3.6's prep kernels (glue.gdn_pre, glue.attn_prep): qwen_prep.cu against every captured Triton variant.

const std = @import("std");
const cuda = @import("cuda");
const qwen27 = @import("qwen27");
const check = @import("check.zig");

const tri = qwen27.triton;
const prep = qwen27.prep_native;

pub const rows_max = 4096;
pub const streams = 16;
pub const keep = 3;
pub const gdn_shape: tri.Gdn = .{ .c = 8192, .kh = 16, .vh = 32, .dk = 128, .keep = keep };
pub const attn_shape: tri.Attn = .{ .heads = 16, .kv_heads = 2, .dim = 256, .half = 32 };

// Every row count the engine serves (decode, lane, verify rows, prompt chunks up to 4096), plus odd ones uncaptured.
const row_counts = blk: {
    var l: [64]usize = undefined;
    var n: usize = 0;
    for (1..33) |m| {
        l[n] = m;
        n += 1;
    }
    for ([_]usize{ 35, 37, 40, 41, 44, 64, 128, 299, 300, 379, 380, 387, 619, 626, 1023, 1024, 1501, 1502, 1509, 2048, 2356, 2357, 2358, 2360, 2361, 3333, 4096 }) |m| {
        l[n] = m;
        n += 1;
    }
    break :blk l[0..n].*;
};

pub fn bf16(v: f32) u16 {
    const u: u32 = @bitCast(v);
    if (std.math.isNan(v)) return 0x7fc0;
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

// A device buffer padded 0xff on both sides; `shifted` puts the data 4 bytes past a 16-byte boundary (Triton's view).
pub const Buf = struct {
    b: cuda.DeviceBuffer,
    off: usize,

    pub fn ptr(x: Buf) u64 {
        return x.b.ptr + x.off;
    }
    pub fn free(x: *Buf) void {
        x.b.free();
    }
};

const Rig = struct {
    gpu: check.Gpu,
    s: cuda.Stream,
    replay: tri.Tri,
    native: tri.Tri,
    cases: usize = 0,
    bad: usize = 0,
    bytes: usize = 0,

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

    // Whole buffers, guard bytes included: an unwritten word, or a write past the end, cannot match.
    fn same(r: *Rig, what: []const u8, case: []const u8, got: Buf, want: Buf) !void {
        try r.s.synchronize();
        const a = try check.download(r.gpu, got.b);
        defer r.gpu.gpa.free(a);
        const w = try check.download(r.gpu, want.b);
        defer r.gpu.gpa.free(w);
        r.cases += 1;
        r.bytes += w.len;
        if (std.mem.eql(u8, a, w)) return;
        r.bad += 1;
        var words: usize = 0;
        var shown: usize = 0;
        var i: usize = 0;
        while (i + 4 <= a.len) : (i += 4) {
            if (std.mem.eql(u8, a[i..][0..4], w[i..][0..4])) continue;
            words += 1;
            if (shown < 3) std.debug.print("    {s} {s}: word {d} native {x:0>8} replay {x:0>8}\n", .{ what, case, i / 4, std.mem.readInt(u32, a[i..][0..4], .little), std.mem.readInt(u32, w[i..][0..4], .little) });
            shown += 1;
        }
        std.debug.print("  DIFF {s} {s}: {d} of {d} words\n", .{ what, case, words, a.len / 4 });
    }
};

pub fn shifted(v: anytype, name: []const u8) bool {
    for (v.spec.params) |p| if (std.mem.eql(u8, p.name, name)) return !p.div16;
    return false;
}

pub fn constOf(v: anytype, name: []const u8) i64 {
    return v.spec.consts.map.get(name).?.int.?;
}

// A random bf16 array: a scale per `block` values from `scales`, a rare outlier of 30 times that.
pub fn fillScaled(rng: std.Random, out: []u16, block: usize, scales: []const f32) void {
    var i: usize = 0;
    while (i < out.len) : (i += block) {
        const sc = scales[rng.uintLessThan(usize, scales.len)];
        for (out[i..@min(out.len, i + block)]) |*v| {
            const spike: f32 = if (rng.uintLessThan(u8, 64) == 0) 30 else 1;
            v.* = bf16(rng.floatNorm(f32) * sc * spike);
        }
    }
}

// dt_bias per head (a ramp), so a case of `a` can cancel it exactly.
fn dtbAt(wild: bool, h: usize) f32 {
    const k: f32 = @floatFromInt(h);
    return (k - 15.5) * (if (wild) @as(f32, 0.45) else 0.07);
}

pub const Gdn = struct {
    qkv: []u16,
    cs: []u16,
    cw: []u16,
    a: []u16,
    b: []u16,
    alog: [32]f32,
    dtb: [32]f32,
    win: [2][]i32, // [kind]: 0 one stream's linear windows (segments for MULTI), 1 random causal windows
    sid: [2][]i32,

    pub fn make(gpa: std.mem.Allocator, rng: std.Random, wild: bool) !Gdn {
        const sc: []const f32 = if (wild) &.{ 0.01, 1, 8, 64, 1000 } else &.{ 0.02, 0.5, 1, 1, 4 };
        var g: Gdn = undefined;
        g.qkv = try gpa.alloc(u16, rows_max * gdn_shape.c);
        g.cs = try gpa.alloc(u16, streams * keep * gdn_shape.c);
        g.cw = try gpa.alloc(u16, gdn_shape.c * (keep + 1));
        g.a = try gpa.alloc(u16, rows_max * gdn_shape.vh);
        g.b = try gpa.alloc(u16, rows_max * gdn_shape.vh);
        fillScaled(rng, g.qkv, gdn_shape.c, sc);
        fillScaled(rng, g.cs, gdn_shape.c, sc);
        fillScaled(rng, g.cw, 4 * 16, if (wild) &.{ 0.1, 0.5, 2 } else &.{ 0.3, 0.5 });
        for (g.a, g.b, 0..) |*a, *b, i| {
            const h = i % gdn_shape.vh;
            // softplus's two branches (20 is the switch), exp's flush range, and sigmoid's saturation
            a.* = switch (rng.uintLessThan(u8, 12)) {
                0 => bf16(@as(f32, 19.5) + rng.float(f32) * 1.5),
                1 => bf16(rng.floatNorm(f32) * 40),
                2 => bf16(-80 - rng.float(f32) * 50),
                3 => bf16(-dtbAt(wild, h)),
                else => bf16(rng.floatNorm(f32) * (if (wild) @as(f32, 6) else 1.5)),
            };
            b.* = switch (rng.uintLessThan(u8, 12)) {
                0 => bf16(rng.floatNorm(f32) * 60),
                1 => bf16(-90 - rng.float(f32) * 40),
                else => bf16(rng.floatNorm(f32) * (if (wild) @as(f32, 8) else 2)),
            };
        }
        for (&g.alog, &g.dtb, 0..) |*al, *db, h| {
            al.* = if (wild) rng.floatNorm(f32) * 3 else @log(1 + rng.float(f32) * 15);
            db.* = dtbAt(wild, h);
        }
        for (&g.win, &g.sid, 0..) |*w, *s, kind| {
            w.* = try gpa.alloc(i32, rows_max * (keep + 1));
            s.* = try gpa.alloc(i32, rows_max);
            var base: usize = 0;
            var seg_end: usize = 0;
            var sid: i32 = 0;
            for (0..rows_max) |r| {
                if (r == seg_end) { // a new stream's rows begin: its state, its own length
                    base = r;
                    seg_end = r + 1 + rng.uintLessThan(usize, 80);
                    sid = @intCast(rng.uintLessThan(usize, streams));
                }
                s.*[r] = if (kind == 0) sid else @intCast(rng.uintLessThan(usize, streams));
                for (0..keep + 1) |j| {
                    w.*[r * (keep + 1) + j] = if (kind == 0)
                        @intCast(if (r - base + j < keep) r - base + j else r + j)
                    else
                        @intCast(rng.uintLessThan(usize, keep + 1 + r)); // state rows, or any row so far
                }
            }
        }
        return g;
    }

    pub fn free(g: *Gdn, gpa: std.mem.Allocator) void {
        for ([_][]u16{ g.qkv, g.cs, g.cw, g.a, g.b }) |x| gpa.free(x);
        for (g.win) |x| gpa.free(x);
        for (g.sid) |x| gpa.free(x);
    }
};

// Native against replay, one gdn_pre entry of the captured set over every row count.
fn gdnVariant(r: *Rig, v: anytype, g: *const Gdn, set: usize) !void {
    const multi = constOf(v, "MULTI") == 1;
    var qkv = try r.up(std.mem.sliceAsBytes(g.qkv), shifted(v, "QKV"));
    defer qkv.free();
    var cs = try r.up(std.mem.sliceAsBytes(if (multi) g.cs else g.cs[0 .. keep * gdn_shape.c]), shifted(v, "CS"));
    defer cs.free();
    var cw = try r.up(std.mem.sliceAsBytes(g.cw), shifted(v, "CW"));
    defer cw.free();
    var a = try r.up(std.mem.sliceAsBytes(g.a), shifted(v, "A"));
    defer a.free();
    var b = try r.up(std.mem.sliceAsBytes(g.b), shifted(v, "B"));
    defer b.free();
    var alog = try r.up(std.mem.sliceAsBytes(&g.alog), shifted(v, "ALOG"));
    defer alog.free();
    var dtb = try r.up(std.mem.sliceAsBytes(&g.dtb), shifted(v, "DTB"));
    defer dtb.free();
    for (0..2) |kind| {
        var win = try r.up(std.mem.sliceAsBytes(g.win[kind]), shifted(v, "WIN"));
        defer win.free();
        var sid = try r.up(std.mem.sliceAsBytes(g.sid[kind]), shifted(v, "SID"));
        defer sid.free();
        for (row_counts) |w| {
            var outs: [2][5]Buf = undefined;
            const sizes = [5]usize{ w * 16 * 128 * 2, w * 16 * 128 * 2, w * 32 * 128 * 2, w * 32 * 4, w * 32 * 4 };
            const names = [5][]const u8{ "Q", "K", "V", "G", "BETA" };
            for (&outs) |*o| for (o, sizes, names) |*x, n, nm| {
                x.* = try r.alloc(n, shifted(v, nm));
            };
            defer for (&outs) |*o| for (o) |*x| x.free();
            for ([_]tri.Tri{ r.replay, r.native }, outs) |t, o| {
                try t.gdnPre(qkv.ptr(), cs.ptr(), cw.ptr(), win.ptr(), if (multi) sid.ptr() else null, a.ptr(), b.ptr(), alog.ptr(), dtb.ptr(), o[0].ptr(), o[1].ptr(), o[2].ptr(), o[3].ptr(), o[4].ptr(), w, gdn_shape);
            }
            var case_buf: [96]u8 = undefined;
            const case = try std.fmt.bufPrint(&case_buf, "{s} rows {d} windows {d} data {d}", .{ v.spec.hash[0..6], w, kind, set });
            for (outs[1], outs[0], names) |got, want, nm| try r.same(nm, case, got, want);
        }
    }
}

pub const Attn = struct {
    qg: []u16,
    kv: []u16,
    qn: [256]u16,
    kn: [256]u16,
    pos: []i32,
    inv: [32]f32,

    pub fn make(gpa: std.mem.Allocator, rng: std.Random, wild: bool) !Attn {
        const sc: []const f32 = if (wild) &.{ 1e-20, 0.01, 1, 30, 1e19 } else &.{ 0.05, 1, 1, 3 };
        var t: Attn = undefined;
        t.qg = try gpa.alloc(u16, rows_max * 16 * 2 * 256);
        t.kv = try gpa.alloc(u16, rows_max * 2 * 256);
        t.pos = try gpa.alloc(i32, rows_max);
        fillScaled(rng, t.qg, 256, sc);
        fillScaled(rng, t.kv, 256, sc);
        for (&t.qn, &t.kn) |*q, *k| {
            q.* = bf16(if (wild) rng.floatNorm(f32) * 3 else 1 + rng.floatNorm(f32) * 0.3);
            k.* = bf16(if (wild) rng.floatNorm(f32) * 3 else 1 + rng.floatNorm(f32) * 0.3);
        }
        // decode runs of consecutive positions, scattered ones, and arguments past the fast range reduction (105615)
        const special = [_]i32{ 0, 1, 8191, 8192, 65536, 1 << 20, 1 << 24, (1 << 24) + 1, 1 << 30, 123456789, std.math.maxInt(i32), -1, -65536 };
        var base: i32 = 0;
        for (t.pos, 0..) |*p, i| {
            if (i % 40 == 0) base = @intCast(rng.uintLessThan(u32, 8192));
            p.* = switch (rng.uintLessThan(u8, 10)) {
                0 => special[rng.uintLessThan(usize, special.len)],
                1, 2 => @intCast(rng.uintLessThan(u32, 8192)),
                else => base + @as(i32, @intCast(i % 40)),
            };
        }
        for (&t.inv, 0..) |*f, i| {
            const e: f64 = @floatFromInt(2 * i);
            f.* = if (wild) rng.float(f32) * 3 else @floatCast(std.math.pow(f64, 1e7, -e / 64));
        }
        return t;
    }

    pub fn free(t: *Attn, gpa: std.mem.Allocator) void {
        gpa.free(t.qg);
        gpa.free(t.kv);
        gpa.free(t.pos);
    }
};

fn attnVariant(r: *Rig, v: anytype, t: *const Attn, set: usize) !void {
    var qg = try r.up(std.mem.sliceAsBytes(t.qg), shifted(v, "QG"));
    defer qg.free();
    var kv = try r.up(std.mem.sliceAsBytes(t.kv), shifted(v, "KV"));
    defer kv.free();
    var qn = try r.up(std.mem.sliceAsBytes(&t.qn), shifted(v, "QN"));
    defer qn.free();
    var kn = try r.up(std.mem.sliceAsBytes(&t.kn), shifted(v, "KN"));
    defer kn.free();
    var pos = try r.up(std.mem.sliceAsBytes(t.pos), shifted(v, "POS"));
    defer pos.free();
    var inv = try r.up(std.mem.sliceAsBytes(&t.inv), shifted(v, "INV"));
    defer inv.free();
    for (row_counts) |w| {
        var outs: [2][2]Buf = undefined;
        const sizes = [2]usize{ w * 16 * 256 * 2, w * 2 * 256 * 2 };
        const names = [2][]const u8{ "QOUT", "KOUT" };
        for (&outs) |*o| for (o, sizes, names) |*x, n, nm| {
            x.* = try r.alloc(n, shifted(v, nm));
        };
        defer for (&outs) |*o| for (o) |*x| x.free();
        for ([_]tri.Tri{ r.replay, r.native }, outs) |tr, o| {
            try tr.attnPrep(qg.ptr(), kv.ptr(), qn.ptr(), kn.ptr(), pos.ptr(), inv.ptr(), o[0].ptr(), o[1].ptr(), w, attn_shape, 1e-6);
        }
        var case_buf: [96]u8 = undefined;
        const case = try std.fmt.bufPrint(&case_buf, "{s} rows {d} data {d}", .{ v.spec.hash[0..6], w, set });
        for (outs[1], outs[0], names) |got, want, nm| try r.same(nm, case, got, want);
    }
}

pub fn run(gpu: check.Gpu, aot_dir: []const u8) !void {
    const gpa = gpu.gpa;
    var set = try cuda.aot.Set.load(gpa, gpu.io, gpu.d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    const s = try cuda.Stream.init(gpu.d, true);
    // TF_QWEN_PREP_FATBIN: another build of the kernels, a control that the data makes a changed kernel differ
    const control: ?[]u8 = if (std.c.getenv("TF_QWEN_PREP_FATBIN")) |p| try std.Io.Dir.cwd().readFileAlloc(gpu.io, std.mem.span(p), gpa, .limited(1 << 26)) else null;
    defer if (control) |c| gpa.free(c);
    var native = if (control) |c| try prep.Native.loadImage(gpu.d, try gpu.ctx.capability(), c) else try prep.Native.load(gpu.d, try gpu.ctx.capability());
    defer native.deinit();
    var r: Rig = .{ .gpu = gpu, .s = s, .replay = .{ .set = &set, .s = s }, .native = .{ .set = &set, .s = s, .prep = &native } };
    var prng = std.Random.DefaultPrng.init(0x7e57_9e9e);
    const rng = prng.random();
    var gdn_entries: usize = 0;
    var attn_entries: usize = 0;
    for ([_]bool{ false, true }, 0..) |wild, ds| {
        var g = try Gdn.make(gpa, rng, wild);
        defer g.free(gpa);
        var t = try Attn.make(gpa, rng, wild);
        defer t.free(gpa);
        for (set.variants) |*v| {
            if (std.mem.eql(u8, v.spec.@"fn", "_gdn_pre")) {
                try gdnVariant(&r, v, &g, ds);
                gdn_entries += 1;
            } else if (std.mem.eql(u8, v.spec.@"fn", "_attn_prep")) {
                try attnVariant(&r, v, &t, ds);
                attn_entries += 1;
            }
        }
        std.debug.print("data set {d}: {d} comparisons so far, {d} differ\n", .{ ds, r.cases, r.bad });
    }
    try check.expect(gdn_entries > 0 and attn_entries > 0, "the captured set has no gdn_pre or attn_prep", .{});
    try check.expect(r.bad == 0, "{d} of {d} comparisons differ", .{ r.bad, r.cases });
    check.pass("qwen-prep: native byte-equal to the captured launches, {d} gdn_pre and {d} attn_prep runs, {d} comparisons, {d} bytes", .{ gdn_entries, attn_entries, r.cases, r.bytes });
}
