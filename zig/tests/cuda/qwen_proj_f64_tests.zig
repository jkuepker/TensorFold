//! Qwen3.6's 4-bit projections (dense, denseGroup, prefillDense) at the served shapes vs host float64.
const std = @import("std");
const cuda = @import("cuda");
const qwen27 = @import("qwen27");
const check = @import("check.zig");
const st = @import("moe_f64_stats.zig");

const kern = qwen27.kernels;
const Gpu = check.Gpu;

const hidden = 2048;
const vocab = 248320;
const decode_rows = [_]usize{ 1, 2, 4, 8, 16, 17, 33, 65, 80 };
const head_rows = [_]usize{ 1, 4, 16 };
const prompt_rows = [_]usize{ 1, 17, 128, 130 };

// fp32 sums on tensor cores: one rounding of the running sum per k16 step, 16 more for the split and group joins.
fn accum(k: usize) f64 {
    return @as(f64, @floatFromInt(k / 16 + 16)) * 0x1p-23;
}

const Mat = struct { words: []u32, scales: []u16, biases: []u16 };
const Ref = struct { want: []f64, tol: []f64 };

const Proj = struct {
    name: []const u8,
    n: usize,
    k: usize,
    q: kern.QLinear,
    m: Mat,
    dec: Ref = undefined,
    pro: Ref = undefined,
};

const Class = struct { name: []const u8, dec: st.Err = .{}, pro: st.Err = .{} };

const Rig = struct {
    gpu: Gpu,
    k: *kern.Kernels,
    s: cuda.Stream,
    ops: kern.Ops,
    rng: std.Random,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    controlled: bool = false,

    fn dev(r: *Rig, bytes: []const u8) !cuda.DeviceBuffer {
        const b = try cuda.DeviceBuffer.fromHost(r.gpu.d, bytes);
        try r.bufs.append(r.gpu.gpa, b);
        return b;
    }

    fn zeros(r: *Rig, n: usize) !cuda.DeviceBuffer {
        const b = try cuda.DeviceBuffer.alloc(r.gpu.d, n);
        try b.fill8(0xff, r.s.handle);
        try r.bufs.append(r.gpu.gpa, b);
        return b;
    }
};

// Random MLX arrays (scales 0.004 to 0.024, biases 4 to 12 scales below zero), tiled as Loader.tile does.
fn make(r: *Rig, a: std.mem.Allocator, name: []const u8, n: usize, k: usize) !Proj {
    const words = try a.alloc(u32, n * k / 8);
    r.rng.bytes(std.mem.sliceAsBytes(words));
    const scales = try a.alloc(u16, n * k / 64);
    const biases = try a.alloc(u16, scales.len);
    for (scales, biases) |*sc, *bi| {
        const s = 0.004 + r.rng.float(f32) * 0.02;
        sc.* = st.bf16(s);
        bi.* = st.bf16(-s * (4 + r.rng.float(f32) * 8));
    }
    const kg = k / 64;
    const sizes = cuda.qlinear.Affine4.layout(n, k);
    const src = [3]cuda.DeviceBuffer{ try r.dev(std.mem.sliceAsBytes(words)), try r.dev(std.mem.sliceAsBytes(scales)), try r.dev(std.mem.sliceAsBytes(biases)) };
    const q: kern.QLinear = .{
        .w = (try r.zeros(sizes.words)).ptr,
        .s = (try r.zeros(sizes.scales)).ptr,
        .b = (try r.zeros(sizes.scales)).ptr,
        .n = n,
        .k = k,
        .npad = sizes.npad,
    };
    const total: u64 = @as(u64, sizes.npad / 64) * kg * 512;
    var g: cuda.Args = .{};
    g.add(src[0].ptr);
    for ([_]usize{ n, k / 8, kg }) |v| g.add(@as(c_int, @intCast(v)));
    g.add(q.w);
    g.add(@as(c_longlong, @intCast(total)));
    try cuda.launch.launch(r.k.pack_dense, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, r.s, &g);
    for ([_]u64{ src[1].ptr, src[2].ptr }, [_]u64{ q.s, q.b }) |from, to| {
        var t: cuda.Args = .{};
        t.add(from);
        for ([_]usize{ n, kg, sizes.npad }) |v| t.add(@as(c_int, @intCast(v)));
        t.add(to);
        const cells = @as(u64, kg) * sizes.npad;
        try cuda.launch.launch(r.k.transpose16, .{ .grid = .{ .x = @intCast((cells + 255) / 256) }, .block = .{ .x = 256 } }, r.s, &t);
    }
    try r.s.synchronize();
    return .{ .name = name, .n = n, .k = k, .q = q, .m = .{ .words = words, .scales = scales, .biases = biases } };
}

fn dot(x: []const f64, w: []const f64) f64 {
    var s = [_]f64{ 0, 0, 0, 0 };
    var i: usize = 0;
    while (i + 4 <= x.len) : (i += 4) {
        inline for (0..4) |j| s[j] += x[i + j] * w[i + j];
    }
    return (s[0] + s[1]) + (s[2] + s[3]);
}

// y = x W^T for `rows` rows of x (values `xf`, magnitudes `xa`), W a row at a time; `rounded`: bf16 weights.
fn reference(a: std.mem.Allocator, p: Proj, xf: []const f64, xa: []const f64, rows: usize, rounded: bool) !Ref {
    const k = p.k;
    const want = try a.alloc(f64, rows * p.n);
    const tol = try a.alloc(f64, rows * p.n);
    const w = try a.alloc(f64, k);
    defer a.free(w);
    const mag = try a.alloc(f64, k);
    defer a.free(mag);
    for (0..p.n) |j| {
        for (0..k) |c| {
            const q: f64 = @floatFromInt((p.m.words[j * (k / 8) + c / 8] >> @intCast(4 * (c % 8))) & 0xF);
            const g = j * (k / 64) + c / 64;
            const s = st.bf(p.m.scales[g]);
            const b = st.bf(p.m.biases[g]);
            const v = if (rounded) st.round(@as(f32, @floatCast(q * s + b))) else q * s + b;
            w[c] = v;
            mag[c] = if (rounded) @abs(v) else q * s + @abs(b);
        }
        for (0..rows) |row| {
            const y = dot(xf[row * k ..][0..k], w);
            want[row * p.n + j] = y;
            tol[row * p.n + j] = st.eps_bf16 * @abs(y) + accum(k) * dot(xa[row * k ..][0..k], mag) + 1e-30;
        }
    }
    return .{ .want = want, .tol = tol };
}

// bf16 rows of a normed hidden state: unit normal, a channel in 256 six times wider.
fn inputs(r: *Rig, a: std.mem.Allocator, rows: usize, k: usize) !struct { x: []u16, xf: []f64, xa: []f64, xs: []f32 } {
    const x = try a.alloc(u16, rows * k);
    for (x, 0..) |*v, i| v.* = st.bf16(r.rng.floatNorm(f32) * (if (i % 256 == 7) @as(f32, 6) else 1));
    const xf = try a.alloc(f64, x.len);
    const xa = try a.alloc(f64, x.len);
    for (xf, xa, x) |*o, *oa, v| {
        o.* = st.bf(v);
        oa.* = @abs(o.*);
    }
    const xs = try a.alloc(f32, rows * k / 64);
    for (xs, 0..) |*v, i| {
        var s: f64 = 0;
        for (xf[i * 64 ..][0..64]) |t| s += t;
        v.* = @floatCast(s);
    }
    return .{ .x = x, .xf = xf, .xa = xa, .xs = xs };
}

fn compare(r: *Rig, a: std.mem.Allocator, label: []const u8, buf: cuda.DeviceBuffer, p: Proj, ref: Ref, rows: usize) !st.Err {
    try r.s.synchronize();
    const raw = try a.alloc(u8, buf.len);
    try buf.download(0, raw);
    const got = try st.decode(a, raw, .bf16, rows * p.n);
    const e = try st.measure(a, got, ref.want[0 .. rows * p.n], ref.tol[0 .. rows * p.n], p.n);
    var name: [160]u8 = undefined;
    try st.ok(try std.fmt.bufPrint(&name, "{s} {s} [{d}, {d}] {d} rows", .{ label, p.name, p.n, p.k, rows }), e);
    if (!r.controlled) {
        r.controlled = true;
        try st.reject(a, "qmm decode", got, ref.want[0 .. rows * p.n], ref.tol[0 .. rows * p.n], p.n);
    }
    return e;
}

fn find(classes: []Class, name: []const u8) *Class {
    for (classes) |*c| if (std.mem.eql(u8, c.name, name)) return c;
    unreachable;
}

// One set of projections on one input: each part alone (dense), fused (denseGroup) and as prompt rows (prefillDense).
fn set(r: *Rig, a: std.mem.Allocator, parts: []Proj, classes: []Class, group: bool) !void {
    const k = parts[0].k;
    const most = decode_rows[decode_rows.len - 1];
    const top = prompt_rows[prompt_rows.len - 1];
    const inp = try inputs(r, a, top, k);
    for (parts) |*p| {
        p.dec = try reference(a, p.*, inp.xf, inp.xa, most, false);
        p.pro = try reference(a, p.*, inp.xf, inp.xa, top, true);
    }
    const x = try r.dev(std.mem.sliceAsBytes(inp.x));
    const xs = try r.dev(std.mem.sliceAsBytes(inp.xs));
    for (decode_rows) |rows| {
        var alone: [4]cuda.DeviceBuffer = undefined;
        var fused: [4]cuda.DeviceBuffer = undefined;
        var items: [4]kern.Item = undefined;
        for (parts, 0..) |p, i| {
            alone[i] = try r.zeros(rows * p.n * 2);
            fused[i] = try r.zeros(rows * p.n * 2);
            items[i] = .{ .q = p.q, .out = fused[i].ptr };
            try r.ops.dense(x.ptr, xs.ptr, p.q, alone[i].ptr, rows);
        }
        if (group) try r.ops.denseGroup(x.ptr, xs.ptr, items[0..parts.len], rows);
        for (parts, 0..) |p, i| {
            const c = find(classes, p.name);
            c.dec = st.worse(c.dec, try compare(r, a, "qmm dense decode", alone[i], p, p.dec, rows));
            if (group) c.dec = st.worse(c.dec, try compare(r, a, "qmm denseGroup decode", fused[i], p, p.dec, rows));
        }
    }
    for (prompt_rows) |rows| {
        for (parts) |p| {
            const out = try r.zeros(rows * p.n * 2);
            try r.ops.prefillDense(x.ptr, p.q, out.ptr, rows);
            const c = find(classes, p.name);
            c.pro = st.worse(c.pro, try compare(r, a, "qmm prefillDense prompt", out, p, p.pro, rows));
        }
    }
}

// lm_head: decode rows only (a prompt pass reads the last row through the same dense), one hidden-sized input.
fn head(r: *Rig, a: std.mem.Allocator, p: *Proj, classes: []Class) !void {
    const rows = head_rows[head_rows.len - 1];
    const inp = try inputs(r, a, rows, hidden);
    const ref = try reference(a, p.*, inp.xf, inp.xa, rows, false);
    const x = try r.dev(std.mem.sliceAsBytes(inp.x));
    const xs = try r.dev(std.mem.sliceAsBytes(inp.xs));
    for (head_rows) |m| {
        const out = try r.zeros(m * p.n * 2);
        try r.ops.dense(x.ptr, xs.ptr, p.q, out.ptr, m);
        const c = find(classes, p.name);
        c.dec = st.worse(c.dec, try compare(r, a, "qmm dense decode", out, p.*, ref, m));
    }
}

/// `kernels`: "native" for the built-in kernels alone, or a captured set's directory.
pub fn run(gpu: Gpu, kernels: []const u8) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var k = try kern.Kernels.load(gpu.gpa, gpu.io, gpu.ctx, if (std.mem.eql(u8, kernels, "native")) null else kernels, 32);
    defer k.deinit();
    var prng = std.Random.DefaultPrng.init(0x9f64);
    var r: Rig = .{ .gpu = gpu, .k = &k, .s = stream, .ops = .{ .k = &k, .s = stream }, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var classes = [_]Class{
        .{ .name = "gdn qkv" }, .{ .name = "gdn z" },  .{ .name = "gdn b" },  .{ .name = "gdn a" },
        .{ .name = "attn q" },  .{ .name = "attn k" }, .{ .name = "attn v" }, .{ .name = "o_proj / out_proj" },
        .{ .name = "lm_head" },
    };
    var gdn = [_]Proj{ try make(&r, a, "gdn qkv", 8192, hidden), try make(&r, a, "gdn z", 4096, hidden), try make(&r, a, "gdn b", 32, hidden), try make(&r, a, "gdn a", 32, hidden) };
    try set(&r, a, &gdn, &classes, true);
    var attn = [_]Proj{ try make(&r, a, "attn q", 8192, hidden), try make(&r, a, "attn k", 512, hidden), try make(&r, a, "attn v", 512, hidden) };
    try set(&r, a, &attn, &classes, true);
    var out = [_]Proj{try make(&r, a, "o_proj / out_proj", hidden, 4096)};
    try set(&r, a, &out, &classes, false);
    var lm = try make(&r, a, "lm_head", vocab, hidden);
    try head(&r, a, &lm, &classes);
    for (classes) |c| {
        var name: [64]u8 = undefined;
        st.table(try std.fmt.bufPrint(&name, "{s} decode", .{c.name}), c.dec, null, "2^-8 |y| + (k/16+16) 2^-23 sum |x| (|q s| + |b|)");
        if (c.pro.count > 0) st.table(try std.fmt.bufPrint(&name, "{s} prompt", .{c.name}), c.pro, null, "as decode, bf16-rounded weights");
    }
    check.pass("qwen-proj-f64: {d} projection classes, decode (dense, denseGroup) and prompt (prefillDense) within their float64 bounds", .{classes.len});
}
