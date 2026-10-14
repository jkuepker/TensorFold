//! Qwen3.6's experts (MLX 4-bit SwiGLU, 256 routed and the shared one, top-8) at the served shapes vs host float64.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const st = @import("moe_f64_stats.zig");

const grouped = cuda.grouped;
const experts = cuda.experts;
const A4 = experts.Affine4Experts;
const Rig = qr.RigT;

const E = 257; // routed experts, then the shared one
const NI = 512; // an expert's inner width
const D = 2048; // hidden
const top_k = 8;
const slots = top_k + 1;

// fp32 sums on tensor cores: one rounding of the running sum per k16 step, 16 more for the split and tile joins.
fn accum(k: usize) f64 {
    return @as(f64, @floatFromInt(k / 16 + 16)) * 0x1p-23;
}

/// One expert stack in MLX layout: words (E, n, k/8), bf16 scales and biases (E, n, k/64).
const Mat = struct { words: []const u32, scales: []const u16, biases: []const u16, n: usize, k: usize };

// Scales and biases like a trained layer's: scale 0.004 to 0.024, bias a few to a dozen scales below zero.
fn mat(r: *Rig, n: usize, k: usize) !Mat {
    const words = try r.a.alloc(u32, E * n * k / 8);
    r.rng.bytes(std.mem.sliceAsBytes(words));
    const scales = try r.a.alloc(u16, E * n * k / 64);
    const biases = try r.a.alloc(u16, scales.len);
    for (scales, biases) |*sc, *bi| {
        const s = 0.004 + r.rng.float(f32) * 0.02;
        sc.* = st.bf16(s);
        bi.* = st.bf16(-s * (4 + r.rng.float(f32) * 8));
    }
    return .{ .words = words, .scales = scales, .biases = biases, .n = n, .k = k };
}

fn mats(r: *Rig, m: Mat) !A4.Mats {
    return .{ .words = try r.dev(std.mem.sliceAsBytes(m.words)), .scales = try r.dev(std.mem.sliceAsBytes(m.scales)), .biases = try r.dev(std.mem.sliceAsBytes(m.biases)) };
}

/// Expert `e`'s (n, k) values in float64 (bf16-rounded for the prompt kernels) and their magnitudes |q s| + |b|.
fn dequant(m: Mat, e: usize, rounded: bool, w: []f64, mag: []f64) void {
    for (0..m.n) |row| {
        for (0..m.k) |col| {
            const q: f64 = @floatFromInt((m.words[(e * m.n + row) * (m.k / 8) + col / 8] >> @intCast(4 * (col % 8))) & 0xF);
            const g = (e * m.n + row) * (m.k / 64) + col / 64;
            const s = st.bf(m.scales[g]);
            const b = st.bf(m.biases[g]);
            const v = if (rounded) st.round(@as(f32, @floatCast(q * s + b))) else q * s + b;
            w[row * m.k + col] = v;
            mag[row * m.k + col] = if (rounded) @abs(v) else q * s + @abs(b);
        }
    }
}

fn dot(x: []const f64, w: []const f64) f64 {
    var s = [_]f64{ 0, 0, 0, 0 };
    var i: usize = 0;
    while (i + 4 <= x.len) : (i += 4) {
        inline for (0..4) |j| s[j] += x[i + j] * w[i + j];
    }
    return (s[0] + s[1]) + (s[2] + s[3]);
}

fn silu(g: f64) f64 {
    return g * st.sigmoid(g);
}

// How far the kernels' SwiGLU can sit from the float64 one: gate and up each a sum error and a bf16 rounding.
fn actBound(g: f64, u: f64, eg: f64, eu: f64) f64 {
    const dg = eg + st.eps_bf16 * @abs(g);
    const du = eu + st.eps_bf16 * @abs(u);
    const sg = silu(g);
    const ds = @max(@abs(silu(g + dg) - sg), @abs(silu(g - dg) - sg)) + st.eps_bf16 * (@abs(sg) + @abs(silu(g + dg) - sg));
    return @abs(u) * ds + du * @abs(sg) + du * ds + st.eps_bf16 * (@abs(u) + du) * (@abs(sg) + ds) + 1e-30;
}

/// Eight routed picks from 256 and the shared expert, a row's slots; `hot`: every row picks the first row's.
fn picks(r: *Rig, rows: usize, hot: bool) ![]i32 {
    const out = try r.a.alloc(i32, rows * slots);
    for (0..rows) |row| {
        if (hot and row > 0) {
            @memcpy(out[row * slots ..][0..slots], out[0..slots]);
            continue;
        }
        var seen: [E - 1]bool = @splat(false);
        for (0..top_k) |j| {
            var e = r.rng.uintLessThan(usize, E - 1);
            while (seen[e]) e = r.rng.uintLessThan(usize, E - 1);
            seen[e] = true;
            out[row * slots + j] = @intCast(e);
        }
        out[row * slots + top_k] = E - 1;
    }
    return out;
}

const Worst = struct { name: []const u8, err: st.Err = .{}, bound: []const u8 };

const Stack = struct { gate: Mat, up: Mat, down: Mat };

// The kernels' answers (activations bf16, y fp32 or bf16) and the float64 reference of both, one batch.
fn batch(r: *Rig, l: experts.Layer, g: experts.Grouped, router: grouped.Router, s: Stack, x: []const u16, rows: usize, prompt: bool, hot: bool, tab: *[4]Worst) !void {
    const pairs = rows * slots;
    const pk = try picks(r, rows, hot);
    const tile = if (prompt) experts.promptTile(l.format) else 16;
    const wide = pairs > grouped.small;
    const p: grouped.Plan = .{
        .members = try r.zeros(pairs * 4),
        .items = try r.zeros(grouped.maxItems(pairs, E, 16) * 12),
        .counts = try r.zeros(8),
        .rank = try r.zeros((if (wide) pairs else 1) * 4),
        .hist = try r.zeros((if (wide) (pairs + 1023) / 1024 * E else 1) * 4),
    };
    try router.route(r.s, try r.dev(std.mem.sliceAsBytes(pk)), pairs, E, tile, p);
    const items = grouped.maxItems(pairs, E, tile);
    const act = try r.zeros(pairs * NI * 2);
    const y = try r.zeros(pairs * D * 4);
    const up_in: experts.Rows = .{ .x = try r.dev(std.mem.sliceAsBytes(x[0 .. rows * D])), .stride = D, .slots = slots };
    const down_in: experts.Rows = .{ .x = act, .stride = NI };
    if (prompt) {
        try g.prompt(r.s, .up, up_in, l, p, act, items, -1);
        try g.prompt(r.s, .down_bf16, down_in, l, p, y, items, -1);
    } else {
        try g.decode(r.s, .up, up_in, l, p, act, items, -1);
        try g.decode(r.s, .down_f32, down_in, l, p, y, items, -1);
    }
    const got_act = try r.back(act, pairs * NI * 2);
    const got_y = try r.back(y, pairs * D * (if (prompt) @as(usize, 2) else 4));
    const xf = try r.a.alloc(f64, rows * D);
    const xa = try r.a.alloc(f64, rows * D);
    for (xf, xa, x[0 .. rows * D]) |*o, *oa, v| {
        o.* = st.bf(v);
        oa.* = @abs(o.*);
    }
    const gw = try r.a.alloc(f64, NI * D);
    const gm = try r.a.alloc(f64, NI * D);
    const uw = try r.a.alloc(f64, NI * D);
    const um = try r.a.alloc(f64, NI * D);
    const dw = try r.a.alloc(f64, D * NI);
    const dm = try r.a.alloc(f64, D * NI);
    const want_a = try r.a.alloc(f64, pairs * NI);
    const tol_a = try r.a.alloc(f64, pairs * NI);
    const want_y = try r.a.alloc(f64, pairs * D);
    const tol_y = try r.a.alloc(f64, pairs * D);
    const act_f = try r.a.alloc(f64, NI);
    const act_m = try r.a.alloc(f64, NI);
    const ky = accum(NI);
    const kx = accum(D);
    for (0..E) |e| {
        var any = false;
        for (pk) |v| any = any or v == e;
        if (!any) continue;
        dequant(s.gate, e, prompt, gw, gm);
        dequant(s.up, e, prompt, uw, um);
        dequant(s.down, e, prompt, dw, dm);
        for (pk, 0..) |v, pr| {
            if (v != e) continue;
            const row = pr / slots;
            for (0..NI) |j| {
                const gv = dot(xf[row * D ..][0..D], gw[j * D ..][0..D]);
                const uv = dot(xf[row * D ..][0..D], uw[j * D ..][0..D]);
                const eg = kx * dot(xa[row * D ..][0..D], gm[j * D ..][0..D]);
                const eu = kx * dot(xa[row * D ..][0..D], um[j * D ..][0..D]);
                want_a[pr * NI + j] = silu(gv) * uv;
                tol_a[pr * NI + j] = actBound(gv, uv, eg, eu);
            }
            // down from the kernel's own bf16 activations: only the down projection is in question
            for (0..NI) |i| {
                act_f[i] = st.bf(std.mem.bytesToValue(u16, got_act[2 * (pr * NI + i) ..][0..2]));
                act_m[i] = @abs(act_f[i]);
            }
            for (0..D) |j| {
                const yv = dot(act_f, dw[j * NI ..][0..NI]);
                want_y[pr * D + j] = yv;
                tol_y[pr * D + j] = ky * dot(act_m, dm[j * NI ..][0..NI]) + (if (prompt) st.eps_bf16 * @abs(yv) else 0) + 1e-30;
            }
        }
    }
    var name: [128]u8 = undefined;
    const label = try std.fmt.bufPrint(&name, "experts {s} {d} rows{s}", .{ if (prompt) "prompt" else "decode", rows, if (hot) ", one hot set" else "" });
    var full: [160]u8 = undefined;
    const ea = try st.measure(r.a, try st.decode(r.a, got_act, .bf16, pairs * NI), want_a, tol_a, NI);
    try st.ok(try std.fmt.bufPrint(&full, "{s} gate-up (SwiGLU)", .{label}), ea);
    const ey = try st.measure(r.a, try st.decode(r.a, got_y, if (prompt) .bf16 else .f32, pairs * D), want_y, tol_y, D);
    try st.ok(try std.fmt.bufPrint(&full, "{s} down", .{label}), ey);
    const idx: usize = if (prompt) 2 else 0;
    tab[idx].err = st.worse(tab[idx].err, ea);
    tab[idx + 1].err = st.worse(tab[idx + 1].err, ey);
    if (rows == 1 and !prompt) {
        try st.reject(r.a, "experts gate-up", try st.decode(r.a, got_act, .bf16, pairs * NI), want_a, tol_a, NI);
        try st.reject(r.a, "experts down", try st.decode(r.a, got_y, .f32, pairs * D), want_y, tol_y, D);
    }
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var prng = std.Random.DefaultPrng.init(0xe4e7);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = stream, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var kmods: [3]cuda.Module = undefined;
    for ([_][]const u8{ cuda.kernels.experts, cuda.kernels.experts_prefill, cuda.kernels.experts_pack }, &kmods) |blob, *m| m.* = try cuda.Module.load(gpu.d, blob);
    defer for (&kmods) |*m| m.unload();
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    const ex = try A4.resolve(kmods[0], kmods[1], kmods[2], sms);
    const router = try grouped.Router.resolve(kmods[0]);
    const s: Stack = .{ .gate = try mat(&r, NI, D), .up = try mat(&r, NI, D), .down = try mat(&r, D, NI) };
    const scratch = try r.zeros(A4.bytes(E, NI, D));
    const up = try r.zeros(2 * A4.bytes(E, NI, D));
    const down = try r.zeros(A4.bytes(E, D, NI));
    try ex.packGateUp(gpu.d, stream, try mats(&r, s.gate), try mats(&r, s.up), scratch, up, E, NI, D);
    const dd = try mats(&r, s.down);
    try ex.pack(stream, dd.words, dd.scales, dd.biases, down, E, D, NI);
    const l: experts.Layer = .{ .format = .affine4, .swiglu = true, .up = up, .down = down, .width = NI, .dims = D, .experts = E };
    const g: experts.Grouped = .{ .affine4 = &ex };
    // normed hidden rows: unit normal, a channel in 256 six times wider
    const x = try r.a.alloc(u16, 128 * D);
    for (x, 0..) |*v, i| v.* = st.bf16(r.rng.floatNorm(f32) * (if (i % 256 == 7) @as(f32, 6) else 1));
    var tab = [4]Worst{
        .{ .name = "experts decode gate-up (SwiGLU act, bf16)", .bound = "bf16 roundings of gate, up, silu, product + fp32 sums" },
        .{ .name = "experts decode down (fp32 pair rows)", .bound = "(512/16+16) 2^-23 sum |a w|" },
        .{ .name = "experts prompt gate-up (SwiGLU act, bf16)", .bound = "as decode, weights rounded to bf16" },
        .{ .name = "experts prompt down (bf16 pair rows)", .bound = "2^-8 |y| + (512/16+16) 2^-23 sum |a w|" },
    };
    for ([_]usize{ 1, 4, 16 }) |rows| try batch(&r, l, g, router, s, x, rows, false, false, &tab);
    try batch(&r, l, g, router, s, x, 96, true, false, &tab);
    try batch(&r, l, g, router, s, x, 128, true, true, &tab);
    for (tab) |w| st.table(w.name, w.err, null, w.bound);
    check.pass("qwen-experts-f64: shared + {d} routed experts, [{d}, {d}] -> [{d}], gate-up and down within their float64 bounds", .{ E - 1, NI, D, D });
}
