//! Qwen3.6's row glue: qwen_glue.cu and embed_rows.cu against the captured Triton launches, one case per aot entry.

const std = @import("std");
const cuda = @import("cuda");
const qwen27 = @import("qwen27");
const check = @import("check.zig");

const glue = qwen27.glue_native;
const Tri = qwen27.triton.Tri;
const Gpu = check.Gpu;
const aot = cuda.aot;

const seed: u64 = 0x1a7e_5eed;
const eps: f32 = 1e-6;
const hidden = glue.hidden;
const vh = glue.value_heads;
const dv = glue.value_dim;
const heads = glue.query_heads;
const hd = glue.head_dim;
const vocab = 4096;

// Row counts the engine serves: decode 1, lane and verify rows 2-16, tree windows, prompt chunks and their remainders.
const serve_rows = [_]usize{
    1,  2,  3,  4,  5,  6,  7,   8,   9,   10,  11,  12,  13,  14,  15,   16,   17,   18,   19,   20,   21,   22,   23,   24,   25, 26, 27, 28, 29, 30, 31, 32,
    35, 37, 40, 41, 44, 64, 128, 299, 300, 379, 380, 387, 619, 626, 1023, 1024, 1501, 1502, 1509, 2356, 2357, 2358, 2360, 2361,
};
const sums_rows = [_]usize{ 1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128 };
const path_rows = [_]usize{
    1,  2,  3,  4,  5,  6,  7,   8,   9,   10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    35, 37, 40, 41, 44, 64, 128, 129, 200,
};

const Kind = enum { normal, outliers, zero, special, subnormal, tiny, huge, wide, spread };

fn bf(f: f32) u16 {
    const u: u32 = @bitCast(f);
    if (f != f) return 0x7fc0;
    return @intCast((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

fn bf32(v: u16) f32 {
    return @bitCast(@as(u32, v) << 16);
}

const Placed = struct { buf: cuda.DeviceBuffer, ptr: u64 };

const Rig = struct {
    gpu: Gpu,
    s: cuda.Stream,
    native: glue.Native,
    replay: Tri,
    prng: std.Random.DefaultPrng,
    pool: [96]cuda.DeviceBuffer = undefined,
    held: usize = 0,
    label: []const u8 = "",
    cases: usize = 0,
    bad: usize = 0,
    total_cases: usize = 0,
    total_bad: usize = 0,
    shown: usize = 0,
    kinds: []Kind = &.{},
    oracle: ?[]const u8 = null, // the Python engine's dumps (qwen36/capture.py): real y rows for group_sums

    fn rand(r: *Rig) std.Random {
        return r.prng.random();
    }

    fn release(r: *Rig) void {
        for (r.pool[0..r.held]) |*b| b.free();
        r.held = 0;
    }

    // Bytes on the device, `ptr` 16-aligned or (for a pointer an entry lists as unaligned) 4 bytes past that.
    fn up(r: *Rig, data: []const u8, aligned: bool) !Placed {
        const pad: usize = if (aligned) 0 else 4;
        const host = try r.gpu.gpa.alloc(u8, data.len + pad + 16);
        defer r.gpu.gpa.free(host);
        @memset(host, 0xa5);
        @memcpy(host[pad..][0..data.len], data);
        const b = try cuda.DeviceBuffer.fromHost(r.gpu.d, host);
        r.pool[r.held] = b;
        r.held += 1;
        return .{ .buf = b, .ptr = b.ptr + pad };
    }

    // An output the kernel must fill whole: 0xff bytes first, so an unwritten or overrun word cannot match.
    fn out(r: *Rig, n: usize) !cuda.DeviceBuffer {
        const b = try cuda.DeviceBuffer.alloc(r.gpu.d, n + 16);
        try b.fill8(0xff, r.s.handle);
        r.pool[r.held] = b;
        r.held += 1;
        return b;
    }

    fn begin(r: *Rig, label: []const u8) void {
        r.label = label;
        r.cases = 0;
        r.bad = 0;
        r.shown = 0;
        std.debug.print("-- {s}\n", .{label});
    }

    fn finish(r: *Rig) void {
        std.debug.print("   {s}: {d} comparisons, {d} differ\n", .{ r.label, r.cases, r.bad });
        r.total_cases += r.cases;
        r.total_bad += r.bad;
    }

    // Whole buffers equal, or the first differing unit (row) and its kind.
    fn same(r: *Rig, part: []const u8, rows: usize, unit_bytes: usize, replay: cuda.DeviceBuffer, native: cuda.DeviceBuffer) !void {
        try r.s.synchronize();
        const ha = try check.download(r.gpu, replay);
        defer r.gpu.gpa.free(ha);
        const hb = try check.download(r.gpu, native);
        defer r.gpu.gpa.free(hb);
        r.cases += 1;
        if (std.mem.eql(u8, ha, hb)) return;
        r.bad += 1;
        if (r.shown >= 12) return;
        r.shown += 1;
        var words: usize = 0;
        var i: usize = 0;
        while (i + 4 <= ha.len) : (i += 4) {
            if (!std.mem.eql(u8, ha[i..][0..4], hb[i..][0..4])) words += 1;
        }
        const first = std.mem.indexOfDiff(u8, ha, hb).?;
        const row = first / unit_bytes;
        const kind: []const u8 = if (row < r.kinds.len) @tagName(r.kinds[row]) else "?";
        const w = first / 4 * 4;
        std.debug.print("  DIFF {s} {s} rows={d}: {d}/{d} words differ, first byte {d} (unit {d}, {s}): replay {x:0>8} native {x:0>8}\n", .{
            r.label, part, rows, words, ha.len / 4, first, row, kind, std.mem.readInt(u32, ha[w..][0..4], .little), std.mem.readInt(u32, hb[w..][0..4], .little),
        });
    }

    // ---- inputs

    fn pickKind(r: *Rig) Kind {
        const v = r.rand().uintLessThan(u32, 100);
        return if (v < 40) .normal else if (v < 50) .outliers else if (v < 55) .zero else if (v < 60) .special else if (v < 65) .subnormal else if (v < 70) .tiny else if (v < 75) .huge else if (v < 82) .wide else if (v < 94) .spread else .normal;
    }

    fn magnitude(r: *Rig) f32 {
        const lo = @log(@as(f32, 0.05));
        const hi = @log(@as(f32, 100.0));
        return @exp(lo + r.rand().float(f32) * (hi - lo));
    }

    fn fillRow(r: *Rig, row: []u16, kind: Kind, scale: f32) void {
        const rng = r.rand();
        const mag = r.magnitude() * scale;
        for (row) |*o| {
            const n = (rng.float(f32) + rng.float(f32) + rng.float(f32) - 1.5) * mag;
            o.* = switch (kind) {
                .normal => bf(n),
                .outliers => bf(if (rng.uintLessThan(u32, 256) == 0) n * 64.0 else n),
                .zero => if (rng.boolean()) 0 else 0x8000,
                .special => bf(n),
                .subnormal => if (rng.uintLessThan(u32, 4) == 0) bf(n) else @as(u16, @intCast(rng.uintLessThan(u32, 0x80))) | (if (rng.boolean()) @as(u16, 0x8000) else 0),
                .tiny => bf(n * 1e-20),
                .huge => bf(n * 1e19),
                .spread => (@as(u16, @intCast(rng.uintLessThan(u32, 80) + 87)) << 7) | @as(u16, @intCast(rng.uintLessThan(u32, 128))) | (if (rng.boolean()) @as(u16, 0x8000) else 0), // exponents 2^-40 to 2^39, random mantissa: nearly every add rounds
                .wide => bf(n * (if (rng.uintLessThan(u32, 8) == 0) @as(f32, 40.0) else @as(f32, 1.0))),
            };
        }
        if (kind == .special) {
            const specials = [_]u16{ 0x7f80, 0xff80, 0x7fc0, 0xffc0, 0x7f81 };
            for (0..1 + rng.uintLessThan(usize, 3)) |_| row[rng.uintLessThan(usize, row.len)] = specials[rng.uintLessThan(usize, specials.len)];
        }
    }

    // `count` vectors of `width` bf16 values, each its own kind and magnitude.
    fn matrix(r: *Rig, count: usize, width: usize, scale: f32) ![]u16 {
        const buf = try r.gpu.gpa.alloc(u16, count * width);
        if (r.kinds.len != 0) r.gpu.gpa.free(r.kinds);
        r.kinds = try r.gpu.gpa.alloc(Kind, count);
        for (0..count) |i| {
            r.kinds[i] = r.pickKind();
            r.fillRow(buf[i * width ..][0..width], r.kinds[i], scale);
        }
        return buf;
    }

    // Norm weights around 1, some negative.
    fn weights(r: *Rig, n: usize) ![]u16 {
        const buf = try r.gpu.gpa.alloc(u16, n);
        const rng = r.rand();
        for (buf) |*o| o.* = bf(1.0 + (rng.float(f32) + rng.float(f32) + rng.float(f32) - 1.5) * 1.2);
        return buf;
    }

    fn reps(rows: usize) usize {
        return if (rows <= 8) 24 else if (rows <= 64) 6 else if (rows <= 300) 2 else 1;
    }
};

fn cint(spec: anytype, name: []const u8) i64 {
    return spec.consts.map.get(name).?.int.?;
}

fn div16(spec: anytype, name: []const u8) bool {
    for (spec.params) |p| if (std.mem.eql(u8, p.name, name)) return p.div16;
    unreachable;
}

fn bytes(s: []const u16) []const u8 {
    return std.mem.sliceAsBytes(s);
}

fn addRms(r: *Rig, spec: anytype) !void {
    if (cint(spec, "D") != hidden or cint(spec, "BLOCK") != hidden) return error.UnexpectedEntry;
    const has_r = cint(spec, "HAS_R") != 0;
    var label: [48]u8 = undefined;
    r.begin(try std.fmt.bufPrint(&label, "_add_rmsnorm[{s}] HAS_R={d}", .{ spec.hash[0..6], @intFromBool(has_r) }));
    const w16 = try r.weights(hidden);
    defer r.gpu.gpa.free(w16);
    for (serve_rows) |rows| for (0..Rig.reps(rows)) |_| {
        defer r.release();
        const res = try r.matrix(rows, hidden, 1.0);
        defer r.gpu.gpa.free(res);
        const x = try r.matrix(rows, hidden, 1.0);
        defer r.gpu.gpa.free(x);
        const xb = try r.up(bytes(x), div16(spec, "X"));
        const rb = try r.up(bytes(res), div16(spec, "R"));
        const wb = try r.up(bytes(w16), div16(spec, "W"));
        const ht = try r.out(rows * hidden * 2);
        const hn = try r.out(rows * hidden * 2);
        const yt = try r.out(rows * hidden * 2);
        const yn = try r.out(rows * hidden * 2);
        const st = try r.out(rows * hidden / 64 * 4);
        const sn = try r.out(rows * hidden / 64 * 4);
        const rp: ?u64 = if (has_r) rb.ptr else null;
        try r.replay.addRmsnorm(xb.ptr, rp, wb.ptr, ht.ptr, yt.ptr, st.ptr, rows, hidden, eps);
        try r.native.addRmsnorm(r.s, xb.ptr, rp, wb.ptr, hn.ptr, yn.ptr, sn.ptr, rows, eps);
        if (has_r) try r.same("H", rows, hidden * 2, ht, hn);
        try r.same("Y", rows, hidden * 2, yt, yn);
        try r.same("XS", rows, hidden / 64 * 4, st, sn);
    };
    r.finish();
}

fn gated(r: *Rig, spec: anytype) !void {
    if (cint(spec, "VH") != vh or cint(spec, "DV") != dv) return error.UnexpectedEntry;
    var label: [48]u8 = undefined;
    r.begin(try std.fmt.bufPrint(&label, "_gated_norm[{s}]", .{spec.hash[0..6]}));
    const w16 = try r.weights(dv);
    defer r.gpu.gpa.free(w16);
    for (serve_rows) |rows| for (0..Rig.reps(rows)) |_| {
        defer r.release();
        const y = try r.matrix(rows * vh, dv, 1.0);
        defer r.gpu.gpa.free(y);
        const z = try r.matrix(rows * vh, dv, 1.5);
        defer r.gpu.gpa.free(z);
        const yb = try r.up(bytes(y), div16(spec, "Yr"));
        const zb = try r.up(bytes(z), div16(spec, "Z"));
        const wb = try r.up(bytes(w16), div16(spec, "W"));
        const ot = try r.out(rows * vh * dv * 2);
        const on = try r.out(rows * vh * dv * 2);
        const st = try r.out(rows * vh * dv / 64 * 4);
        const sn = try r.out(rows * vh * dv / 64 * 4);
        try r.replay.gatedNorm(yb.ptr, zb.ptr, wb.ptr, ot.ptr, st.ptr, rows, vh, dv, eps);
        try r.native.gatedNorm(r.s, yb.ptr, zb.ptr, wb.ptr, on.ptr, sn.ptr, rows, vh, eps);
        try r.same("OUT", rows, dv * 2, ot, on);
        try r.same("XS", rows, dv / 64 * 4, st, sn);
    };
    r.finish();
}

fn gateMul(r: *Rig, spec: anytype) !void {
    if (cint(spec, "H") != heads or cint(spec, "D") != hd) return error.UnexpectedEntry;
    var label: [48]u8 = undefined;
    r.begin(try std.fmt.bufPrint(&label, "_gate_mul[{s}]", .{spec.hash[0..6]}));
    for (serve_rows) |rows| for (0..Rig.reps(rows)) |_| {
        defer r.release();
        const o = try r.matrix(rows * heads, hd, 1.0);
        defer r.gpu.gpa.free(o);
        const qg = try r.matrix(rows * heads * 2, hd, 1.5); // [q | gate] rows: the gate half is the odd vectors
        defer r.gpu.gpa.free(qg);
        const ob = try r.up(bytes(o), div16(spec, "O"));
        const qb = try r.up(bytes(qg), div16(spec, "QG"));
        const ot = try r.out(rows * heads * hd * 2);
        const on = try r.out(rows * heads * hd * 2);
        const st = try r.out(rows * heads * hd / 64 * 4);
        const sn = try r.out(rows * heads * hd / 64 * 4);
        try r.replay.gateMul(ob.ptr, qb.ptr, ot.ptr, st.ptr, rows, heads, hd);
        try r.native.gateMul(r.s, ob.ptr, qb.ptr, on.ptr, sn.ptr, rows, heads);
        try r.same("OUT", rows, hd * 2, ot, on);
        try r.same("XS", rows, hd / 64 * 4, st, sn);
    };
    r.finish();
}

fn groupSums(r: *Rig, spec: anytype) !void {
    const k: usize = @intCast(cint(spec, "KG") * cint(spec, "GS"));
    var label: [48]u8 = undefined;
    r.begin(try std.fmt.bufPrint(&label, "_group_sums[{s}] K={d}", .{ spec.hash[0..6], k }));
    // ldx the entry lists as a multiple of 16: the row width itself, a padded row, and a wide strided head row
    for ([_]usize{ k, k + 16, 2 * k + 48 }) |ld| for (sums_rows) |rows| for (0..Rig.reps(rows)) |_| {
        defer r.release();
        const x = try r.matrix(rows, ld, 1.0);
        defer r.gpu.gpa.free(x);
        const xb = try r.up(bytes(x), div16(spec, "X"));
        const st = try r.out(rows * (k / 64) * 4);
        const sn = try r.out(rows * (k / 64) * 4);
        try r.replay.groupSums(xb.ptr, st.ptr, ld, rows, k);
        try r.native.groupSums(r.s, xb.ptr, sn.ptr, ld, rows, k);
        try r.same("XS", rows, k / 64 * 4, st, sn);
    };
    r.finish();
}

// Real rows: every dumped y (NNN_y.bin) of the oracle's steps and prefill chunks through group_sums, replay vs native.
fn realSums(r: *Rig, spec: anytype, oracle: []const u8) !void {
    const k: usize = @intCast(cint(spec, "KG") * cint(spec, "GS"));
    var label: [48]u8 = undefined;
    r.begin(try std.fmt.bufPrint(&label, "_group_sums[{s}] real y rows", .{spec.hash[0..6]}));
    var files: usize = 0;
    if (r.kinds.len != 0) r.gpu.gpa.free(r.kinds);
    r.kinds = &.{};
    for ([_][]const u8{ "story", "code", "facts", "long" }) |prompt| for ([_][]const u8{ "cold", "warm", "prefill" }) |phase| {
        var step: usize = 0;
        while (step < 4096) : (step += 1) {
            var dir_buf: [256]u8 = undefined;
            const dir = if (std.mem.eql(u8, phase, "prefill")) try std.fmt.bufPrint(&dir_buf, "{s}/{s}/{s}", .{ oracle, prompt, phase }) else try std.fmt.bufPrint(&dir_buf, "{s}/{s}/{s}/step{d:0>3}", .{ oracle, prompt, phase, step });
            var n: usize = 0;
            while (n < 512) : (n += 1) {
                defer r.release();
                var path_buf: [320]u8 = undefined;
                const path = try std.fmt.bufPrint(&path_buf, "{s}/{d:0>3}_y.bin", .{ dir, n });
                const data = std.Io.Dir.cwd().readFileAlloc(r.gpu.io, path, r.gpu.gpa, .limited(1 << 30)) catch |e| switch (e) {
                    error.FileNotFound => break,
                    else => return e,
                };
                defer r.gpu.gpa.free(data);
                const rows = data.len / (k * 2);
                if (rows == 0 or data.len % (k * 2) != 0) return error.UnexpectedDump;
                const xb = try r.up(data, true);
                const st = try r.out(rows * (k / 64) * 4);
                const sn = try r.out(rows * (k / 64) * 4);
                try r.replay.groupSums(xb.ptr, st.ptr, k, rows, k);
                try r.native.groupSums(r.s, xb.ptr, sn.ptr, k, rows, k);
                try r.same("XS", rows, k / 64 * 4, st, sn);
                files += 1;
            }
            if (n == 0 or std.mem.eql(u8, phase, "prefill")) break;
        }
    };
    std.debug.print("   {d} dumped y files\n", .{files});
    r.finish();
}

// A forest of `rows` nodes: parents below the child (-1 at roots), shaped by `shape`.
fn forest(r: *Rig, parents: []i32, shape: usize) void {
    const rng = r.rand();
    for (parents, 0..) |*p, i| {
        p.* = switch (shape) {
            0 => if (i == 0) -1 else @as(i32, @intCast(i - 1)), // one chain
            1 => if (i == 0 or rng.uintLessThan(u32, 8) == 0) -1 else @as(i32, @intCast(rng.uintLessThan(usize, i))), // bushy, a few roots
            2 => -1, // all roots
            3 => if (i == 0) -1 else @as(i32, @intCast((i - 1) / 2)), // binary heap
            else => if (i < 4 or rng.uintLessThan(u32, 3) == 0) (if (i == 0) -1 else @as(i32, @intCast(i - 1))) else @as(i32, @intCast(i - 1 - rng.uintLessThan(usize, @min(i, 3)))), // long chains with stray branches
        };
    }
}

fn paths(r: *Rig, spec: anytype) !void {
    if (cint(spec, "MAXD") != glue.max_depth) return error.UnexpectedEntry;
    var label: [48]u8 = undefined;
    r.begin(try std.fmt.bufPrint(&label, "_paths[{s}]", .{spec.hash[0..6]}));
    for (path_rows) |rows| for (0..5) |shape| for (0..4) |_| {
        defer r.release();
        const parents = try r.gpu.gpa.alloc(i32, rows);
        defer r.gpu.gpa.free(parents);
        forest(r, parents, shape);
        const pb = try r.up(std.mem.sliceAsBytes(parents), div16(spec, "PARENTS"));
        const pt = try r.out(rows * glue.max_depth * 4);
        const pn = try r.out(rows * glue.max_depth * 4);
        const dt = try r.out(rows * 4);
        const dn = try r.out(rows * 4);
        try r.replay.paths(pb.ptr, pt.ptr, dt.ptr, rows);
        try r.native.paths(r.s, pb.ptr, pn.ptr, dn.ptr, rows);
        try r.same("PATHS", rows, glue.max_depth * 4, pt, pn);
        try r.same("DEPTHS", rows, 4, dt, dn);
    };
    r.finish();
}

fn embed(r: *Rig, spec: anytype) !void {
    const d: usize = @intCast(cint(spec, "D"));
    var label: [48]u8 = undefined;
    r.begin(try std.fmt.bufPrint(&label, "_embed[{s}]", .{spec.hash[0..6]}));
    const rng = r.rand();
    const words = try r.gpu.gpa.alloc(u32, vocab * d / 8);
    defer r.gpu.gpa.free(words);
    for (words) |*w| w.* = rng.int(u32);
    const sc = try r.gpu.gpa.alloc(u16, vocab * d / 64);
    defer r.gpu.gpa.free(sc);
    const bi = try r.gpu.gpa.alloc(u16, vocab * d / 64);
    defer r.gpu.gpa.free(bi);
    for (sc) |*o| o.* = bf((rng.float(f32) + 0.05) * 0.1);
    for (bi) |*o| o.* = bf((rng.float(f32) - 0.5) * 2.0);
    var tables = std.ArrayList(cuda.DeviceBuffer).empty;
    defer tables.deinit(r.gpu.gpa);
    defer for (tables.items) |*b| b.free();
    const wt = try cuda.DeviceBuffer.fromHost(r.gpu.d, std.mem.sliceAsBytes(words));
    try tables.append(r.gpu.gpa, wt);
    const st = try cuda.DeviceBuffer.fromHost(r.gpu.d, bytes(sc));
    try tables.append(r.gpu.gpa, st);
    const bt = try cuda.DeviceBuffer.fromHost(r.gpu.d, bytes(bi));
    try tables.append(r.gpu.gpa, bt);
    for (serve_rows) |rows| for (0..Rig.reps(rows)) |_| {
        defer r.release();
        const ids = try r.gpu.gpa.alloc(i32, rows);
        defer r.gpu.gpa.free(ids);
        for (ids) |*x| x.* = @intCast(rng.uintLessThan(u32, vocab));
        ids[0] = if (rng.boolean()) 0 else vocab - 1;
        const ib = try r.up(std.mem.sliceAsBytes(ids), div16(spec, "IDS"));
        const ot = try r.out(rows * d * 2);
        const on = try r.out(rows * d * 2);
        try r.replay.embed(ib.ptr, wt.ptr, st.ptr, bt.ptr, ot.ptr, rows, d);
        try r.native.embed(r.s, ib.ptr, wt.ptr, st.ptr, bt.ptr, on.ptr, rows, d);
        try r.same("OUT", rows, d * 2, ot, on);
    };
    r.finish();
}

pub fn run(gpu: Gpu, aot_dir: []const u8, oracle: ?[]const u8) !void {
    var set = try aot.Set.load(gpu.gpa, gpu.io, gpu.d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    var native = try glue.Native.load(gpu.d, try gpu.ctx.capability());
    defer native.deinit();
    var r: Rig = .{ .gpu = gpu, .s = try cuda.Stream.init(gpu.d, true), .native = native, .replay = undefined, .prng = std.Random.DefaultPrng.init(seed) };
    defer if (r.kinds.len != 0) gpu.gpa.free(r.kinds);
    r.replay = .{ .set = &set, .s = r.s, .glue = null };
    var entries: usize = 0;
    for (set.variants) |v| {
        const name = v.spec.@"fn";
        if (std.mem.eql(u8, name, "_add_rmsnorm")) try addRms(&r, v.spec) else if (std.mem.eql(u8, name, "_gated_norm")) try gated(&r, v.spec) else if (std.mem.eql(u8, name, "_gate_mul")) try gateMul(&r, v.spec) else if (std.mem.eql(u8, name, "_group_sums")) {
            try groupSums(&r, v.spec);
            if (oracle) |dir| try realSums(&r, v.spec, dir);
        } else if (std.mem.eql(u8, name, "_paths")) try paths(&r, v.spec) else if (std.mem.eql(u8, name, "_embed")) try embed(&r, v.spec) else continue;
        entries += 1;
    }
    std.debug.print("RESULT qwen-glue: {d} entries, {d} comparisons, {d} differ\n", .{ entries, r.total_cases, r.total_bad });
    if (r.total_bad != 0) return error.TestFailed;
}
