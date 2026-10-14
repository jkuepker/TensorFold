//! Qwen3.6 tree attention: qwen_attention.cu's _shared, _tail and _merge against captured Triton cubins, byte-equal.

const std = @import("std");
const cuda = @import("cuda");
const qwen27 = @import("qwen27");
const check = @import("check.zig");

const aot = cuda.aot;
const kern = qwen27.kernels;
const st = qwen27.state;
const tri = qwen27.triton;
const native = qwen27.tree_native;
const round = qwen27.round;

pub const Tri = tri.Tri;
pub const H = 16;
pub const HK = 2;
pub const D = 256;
pub const kv_row = HK * D; // elements a cached position holds
pub const capacity = 8192 + 256;
const nan_words: u32 = 0x7fc17fc1; // two bf16 NaNs
pub const geometry: qwen27.config.Geometry = .{ .hidden = 2048, .intermediate = 0, .layers = 4, .linear_k_heads = 16, .linear_v_heads = 32, .query_heads = H, .kv_heads = HK, .vocab = 4096 };
pub const attn: tri.Attn = .{ .heads = H, .kv_heads = HK, .dim = D, .half = 32 };
pub const committed = [_]usize{ 0, 1, 511, 512, 513, 1023, 1024, 2047, 2048, 2049, 4095, 4096, 6143, 6144, 8191 };
pub const widths = [_]usize{ 1, 2, 3, 4, 8, 16, 17, 32, 44, 64, 128 };
pub const lane_widths = [_]usize{ 1, 2, 3, 4, 8, 16 };
pub const zero_tokens: [128]u32 = @splat(0);

pub const Kind = enum { chain, bushy };
pub const Spec = struct { p: usize, w: usize, kind: Kind = .chain, k: u8 = 0, v: u8 = 0 }; // k, v: the caches start 8 elements times these past the stream's first rows

pub fn bf16(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

pub const Buffers = struct {
    o: cuda.DeviceBuffer,
    m: cuda.DeviceBuffer,
    l: cuda.DeviceBuffer,

    pub fn parts(b: Buffers) tri.Tri.Partials {
        return .{ .o = b.o.ptr, .m = b.m.ptr, .l = b.l.ptr };
    }
};

pub const Rig = struct {
    gpu: check.Gpu,
    s: cuda.Stream,
    k: *kern.Kernels,
    scratch: *st.Scratch,
    seqs: []st.Seq,
    home: [st.max_streams][2]u64, // each stream's attention-layer key and value caches as the arena gave them (the first `seqs.len`)
    cap: usize, // positions each stream's caches hold
    pool: [2][2]cuda.DeviceBuffer, // pristine keys and values: normal, heavy tail
    q: cuda.DeviceBuffer,
    kn: cuda.DeviceBuffer,
    vn: cuda.DeviceBuffer,
    out: [2]cuda.DeviceBuffer, // replay, native
    part: [2]Buffers,
    nat: native.Native,
    prng: std.Random.DefaultPrng,
    covered: []bool,
    cases: usize = 0,
    comparisons: usize = 0,
    bytes: usize = 0,
    bad: usize = 0,
    skipped: usize = 0,
    written: usize = 0, // words the replay wrote (not the 0xff fill), over all comparisons

    pub fn ops(r: *Rig) kern.Ops {
        return .{ .k = r.k, .s = r.s };
    }

    pub fn replay(r: *Rig) Tri {
        return .{ .set = &r.k.triton.?, .s = r.s };
    }

    // The Tri that stages a round: Triton's launches where a set is loaded, else the built-in row glue alone.
    pub fn staging(r: *Rig) Tri {
        return if (r.k.triton) |*t| .{ .set = t, .s = r.s } else .{ .set = null, .s = r.s, .glue = if (r.k.glue) |*g| g else null };
    }

    pub fn fresh(r: *Rig, n: usize, heavy: bool, scale: f32) !cuda.DeviceBuffer {
        const rng = r.prng.random();
        const host = try r.gpu.gpa.alloc(u16, n);
        defer r.gpu.gpa.free(host);
        for (host) |*x| {
            var v = rng.floatNorm(f32) * scale;
            if (heavy and rng.uintLessThan(u8, 40) == 0) v *= 12;
            x.* = bf16(v);
        }
        return cuda.DeviceBuffer.fromHost(r.gpu.d, std.mem.sliceAsBytes(host));
    }

    // Fills the 0xff pattern a kernel must leave in every slot it does not write.
    pub fn clear(r: *Rig, b: Buffers) !void {
        try b.o.fill8(0xff, r.s.handle);
        try b.m.fill8(0xff, r.s.handle);
        try b.l.fill8(0xff, r.s.handle);
    }

    // Each stream's caches (OFF a multiple of 8 elements): pristine rows below its committed length, NaN words above.
    pub fn load(r: *Rig, specs: []const Spec, pool: usize) !void {
        const o = r.ops();
        for (specs, 0..) |sp, i| {
            const seq = &r.seqs[i];
            seq.pos = sp.p;
            seq.keys[3] = r.home[i][0] + @as(u64, sp.k) * 16;
            seq.values[3] = r.home[i][1] + @as(u64, sp.v) * 16;
            const slack = i * 4096 * 2;
            for ([_]struct { dst: u64, src: cuda.DeviceBuffer }{ .{ .dst = seq.keys[3], .src = r.pool[pool][0] }, .{ .dst = seq.values[3], .src = r.pool[pool][1] } }) |c| {
                try o.copy(c.dst, c.src.ptr + slack, sp.p * kv_row * 2);
                try o.fill32(c.dst + sp.p * kv_row * 2, nan_words, (r.cap - 128 - sp.p) * kv_row / 2); // the 128 spare positions hold the shift
            }
        }
    }

    // The same bytes in the compared prefix of both buffers, else the first differences.
    pub fn same(r: *Rig, what: []const u8, name: []const u8, a: cuda.DeviceBuffer, b: cuda.DeviceBuffer, bytes: usize) !void {
        try r.s.synchronize();
        const gpa = r.gpu.gpa;
        const x = try gpa.alloc(u8, bytes);
        defer gpa.free(x);
        const y = try gpa.alloc(u8, bytes);
        defer gpa.free(y);
        try a.download(0, x);
        try b.download(0, y);
        r.comparisons += 1;
        r.bytes += bytes;
        var j: usize = 0;
        while (j + 4 <= bytes) : (j += 4) r.written += @intFromBool(!std.mem.eql(u8, x[j..][0..4], &.{ 0xff, 0xff, 0xff, 0xff }));
        if (std.mem.eql(u8, x, y)) return;
        r.bad += 1;
        var words: usize = 0;
        var i: usize = 0;
        while (i + 4 <= bytes) : (i += 4) words += @intFromBool(!std.mem.eql(u8, x[i..][0..4], y[i..][0..4]));
        std.debug.print("  DIFFER {s} {s}: {d} of {d} words\n", .{ what, name, words, bytes / 4 });
        var shown: usize = 0;
        i = 0;
        while (i + 4 <= bytes and shown < 3) : (i += 4) {
            if (std.mem.eql(u8, x[i..][0..4], y[i..][0..4])) continue;
            std.debug.print("    word {d}: replay {x:0>8} native {x:0>8}\n", .{ i / 4, std.mem.readInt(u32, x[i..][0..4], .little), std.mem.readInt(u32, y[i..][0..4], .little) });
            shown += 1;
        }
    }

    pub fn partials(r: *Rig, name: []const u8, slots: usize, rows: usize) !void {
        const unit = slots * rows * H;
        const guard = 4096; // bytes past the prefix: a stray write shows
        try r.same("PO", name, r.part[0].o, r.part[1].o, @min(unit * D * 4 + guard, r.part[0].o.len));
        try r.same("PM", name, r.part[0].m, r.part[1].m, @min(unit * 4 + guard, r.part[0].m.len));
        try r.same("PL", name, r.part[0].l, r.part[1].l, @min(unit * 4 + guard, r.part[0].l.len));
    }
};

// Parents of a window (one root, as the engine builds them): a chain, or a random tree of short branches.
pub fn parentsOf(rng: std.Random, kind: Kind, out: []i32) void {
    for (out, 0..) |*x, i| x.* = switch (kind) {
        .chain => @as(i32, @intCast(i)) - 1,
        .bushy => if (i == 0) -1 else if (rng.uintLessThan(u8, 10) < 7) @as(i32, @intCast(i)) - 1 else @intCast(i - 1 - rng.uintLessThan(usize, @min(i, 8))),
    };
}

pub const Staged = struct { r: round.Round, rows: usize, slots: usize };

pub fn stage(rig: *Rig, specs: []const Spec, par: *[16][128]i32) !Staged {
    var parts: [16]round.Part = undefined;
    var slots: usize = 0;
    var rows: usize = 0;
    for (specs, 0..) |sp, i| {
        parentsOf(rig.prng.random(), sp.kind, par[i][0..sp.w]);
        parts[i] = .{ .seq = &rig.seqs[i], .tokens = zero_tokens[0..sp.w], .parents = if (sp.kind == .chain) null else par[i][0..sp.w] };
        slots = @max(slots, tri.slots(sp.p, sp.w));
        rows += sp.w;
    }
    const r = try round.stage(rig.gpu.gpa, rig.ops(), rig.staging(), rig.scratch, geometry, parts[0..specs.len]);
    return .{ .r = r, .rows = rows, .slots = slots };
}

pub fn inputs(rig: *Rig, heavy: bool) !void {
    for ([_]*cuda.DeviceBuffer{ &rig.q, &rig.kn, &rig.vn }, 0..) |b, i| {
        b.free();
        const heads: usize = if (i == 0) H else HK;
        b.* = try rig.fresh(st.round_rows * heads * D, heavy, if (heavy and i == 0) 2.0 else 1.0);
    }
}

pub fn label(buf: []u8, specs: []const Spec, pool: usize) []const u8 {
    var w = std.Io.Writer.fixed(buf);
    w.print("pool {d}:", .{pool}) catch {};
    for (specs[0..@min(specs.len, 4)]) |sp| w.print(" p{d}w{d}{s}", .{ sp.p, sp.w, @tagName(sp.kind)[0..1] }) catch {};
    if (specs.len > 4) w.print(" +{d}", .{specs.len - 4}) catch {};
    return w.buffered();
}

// The whole attention call on one plan, replay against native.
fn whole(rig: *Rig, specs: []const Spec, pool: usize) !void {
    var par: [16][128]i32 = undefined;
    try rig.load(specs, pool);
    try inputs(rig, pool == 1);
    const sg = try stage(rig, specs, &par);
    var lb: [160]u8 = undefined;
    const name = label(&lb, specs, pool);
    for (rig.part) |b| try rig.clear(b);
    for (rig.out) |b| try b.fill8(0xff, rig.s.handle);
    const offs = sg.r.offsets(3, 0);
    rig.cases += 1;
    var t = rig.replay();
    t.attention(rig.q.ptr, rig.kn.ptr, rig.vn.ptr, rig.scratch.origin, offs, sg.r.attn, rig.part[0].parts(), rig.out[0].ptr, attn) catch |e| switch (e) {
        error.MissingTritonVariant => {
            rig.skipped += 1;
            return;
        },
        else => return e,
    };
    t.tree = rig.nat;
    try t.attention(rig.q.ptr, rig.kn.ptr, rig.vn.ptr, rig.scratch.origin, offs, sg.r.attn, rig.part[1].parts(), rig.out[1].ptr, attn);
    try rig.partials(name, sg.slots, sg.r.attn.width);
    try rig.same("OUT", name, rig.out[0], rig.out[1], sg.rows * H * D * 2);
}

fn run1(rig: *Rig, specs: []const Spec, pool: usize) void {
    whole(rig, specs, pool) catch |e| {
        var lb: [160]u8 = undefined;
        std.debug.print("  case failed: {t} ({s})\n", .{ e, label(&lb, specs, pool) });
        rig.bad += 1;
    };
}

fn grid(rig: *Rig) void {
    var n: usize = 0;
    for (committed) |p| for (widths) |w| {
        if (p + w > capacity) continue;
        const shift: u8 = @intCast(n % 5);
        run1(rig, &.{.{ .p = p, .w = w, .k = shift, .v = (shift * 3) % 5 }}, n % 4 / 3); // every fourth on the heavy-tail pool
        run1(rig, &.{.{ .p = p, .w = w, .kind = .bushy, .k = (shift + 2) % 5, .v = shift }}, 0);
        n += 1;
    };
}

fn lanes(rig: *Rig, count: usize) void {
    const rng = rig.prng.random();
    var specs: [16]Spec = undefined;
    for (0..count) |_| {
        const s = 2 + rng.uintLessThan(usize, 15);
        for (specs[0..s]) |*sp| sp.* = .{ .p = committed[rng.uintLessThan(usize, committed.len)], .w = lane_widths[rng.uintLessThan(usize, lane_widths.len)], .kind = if (rng.boolean()) .chain else .bushy, .k = rng.uintLessThan(u8, 5), .v = rng.uintLessThan(u8, 5) };
        run1(rig, specs[0..s], rng.uintLessThan(usize, 2));
    }
}

// ---- one captured variant at a time: its pointers' alignment and W as its aot.json entry lists them

fn relocate(rig: *Rig, keep: *std.ArrayList(cuda.DeviceBuffer), src: u64, bytes: usize, misaligned: bool) !u64 {
    var b = try cuda.DeviceBuffer.alloc(rig.gpu.d, bytes + 64);
    errdefer b.free();
    try keep.append(rig.gpu.gpa, b);
    const dst = b.ptr + @as(u64, if (misaligned) 4 else 0);
    try rig.ops().copy(dst, src, bytes);
    return dst;
}

fn param(v: anytype, name: []const u8) ?bool {
    for (v.spec.params) |p| if (std.mem.eql(u8, p.name, name)) return p.div16;
    return null;
}

fn variantArgs(comptime fname: []const u8, plan: tri.Tri.Plan, q: u64, kn: u64, vn: u64, base: u64, offs: u64, part: tri.Tri.Partials, out: u64, out_args: *[16]aot.Arg) []const aot.Arg {
    const p = aot.ptr;
    const w: i32 = @intCast(plan.width);
    const a = out_args;
    if (comptime std.mem.eql(u8, fname, "_shared")) {
        a[0..10].* = .{ p("Q", "*bf16", q), p("KC", "*bf16", base), p("VC", "*bf16", base), p("OFF", "*i64", offs), p("STREAM", "*i32", plan.streams), p("ITEMS", "*i32", plan.items), p("PO", "*fp32", part.o), p("PM", "*fp32", part.m), p("PL", "*fp32", part.l), aot.int("W", w) };
        return a[0..10];
    }
    if (comptime std.mem.eql(u8, fname, "_tail")) {
        a[0..14].* = .{ p("Q", "*bf16", q), p("KN", "*bf16", kn), p("VN", "*bf16", vn), p("KC", "*bf16", base), p("VC", "*bf16", base), p("OFF", "*i64", offs), p("STREAM", "*i32", plan.streams), p("ROWS", "*i32", plan.rows), p("PATHS", "*i32", plan.paths), p("DEPTHS", "*i32", plan.depths), p("PO", "*fp32", part.o), p("PM", "*fp32", part.m), p("PL", "*fp32", part.l), aot.int("W", w) };
        return a[0..14];
    }
    a[0..7].* = .{ p("PO", "*fp32", part.o), p("PM", "*fp32", part.m), p("PL", "*fp32", part.l), p("OUT", "*bf16", out), p("STREAM", "*i32", plan.streams), p("ROWS", "*i32", plan.rows), aot.int("W", w) };
    return a[0..7];
}

fn variant(rig: *Rig, vi: usize) !void {
    const v = rig.k.triton.?.variants[vi];
    const fname = v.spec.@"fn";
    const is_shared = std.mem.eql(u8, fname, "_shared");
    const is_tail = std.mem.eql(u8, fname, "_tail");
    const is_merge = std.mem.eql(u8, fname, "_merge");
    if (!is_shared and !is_tail and !is_merge) return;
    const w: usize = if (param(v, "W")) |d16| (if (d16) 16 else 3) else 1;
    const specs = [_]Spec{.{ .p = 1536, .w = w, .kind = if (w == 1) .chain else .bushy }};
    var par: [16][128]i32 = undefined;
    try rig.load(&specs, 0);
    try inputs(rig, false);
    const sg = try stage(rig, &specs, &par);
    var keep: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer {
        for (keep.items) |*b| b.free();
        keep.deinit(rig.gpu.gpa);
    }
    try rig.s.synchronize();
    var plan = sg.r.attn;
    plan.streams = try relocate(rig, &keep, plan.streams, 16, !(param(v, "STREAM") orelse true));
    if (is_shared) plan.items = try relocate(rig, &keep, plan.items, plan.n_items * 12, !(param(v, "ITEMS") orelse true));
    plan.rows = try relocate(rig, &keep, plan.rows, w * 4, !(param(v, "ROWS") orelse true));
    try rig.s.synchronize();
    const offs = sg.r.offsets(3, 0);
    var arg_buf: [16]aot.Arg = undefined;
    const parts = rig.part[0].parts();
    const found = if (is_shared) try rig.k.triton.?.find("_shared", variantArgs("_shared", plan, rig.q.ptr, rig.kn.ptr, rig.vn.ptr, rig.scratch.origin, offs, parts, rig.out[0].ptr, &arg_buf), &.{
        aot.ci("H", H), aot.ci("HK", HK), aot.ci("D", D), aot.ci("G", H / HK), aot.ci("CH", tri.chunk), aot.cf("SCALE", 0.0625), aot.ci("GR", tri.group_chunks),
    }) else if (is_tail) try rig.k.triton.?.find("_tail", variantArgs("_tail", plan, rig.q.ptr, rig.kn.ptr, rig.vn.ptr, rig.scratch.origin, offs, parts, rig.out[0].ptr, &arg_buf), &.{
        aot.ci("H", H), aot.ci("HK", HK), aot.ci("D", D), aot.ci("G", H / HK), aot.ci("CH", tri.chunk), aot.ci("MAXD", tri.max_nodes), aot.cf("SCALE", 0.0625), aot.ci("GR", tri.group_chunks),
    }) else try rig.k.triton.?.find("_merge", variantArgs("_merge", plan, rig.q.ptr, rig.kn.ptr, rig.vn.ptr, rig.scratch.origin, offs, parts, rig.out[0].ptr, &arg_buf), &.{
        aot.ci("H", H), aot.ci("D", D), aot.ci("G", H / HK), aot.ci("DS", 64), aot.ci("CH", tri.chunk), aot.ci("GR", tri.group_chunks),
    });
    try check.expect(std.mem.eql(u8, found.spec.hash, v.spec.hash), "variant {s} {s} resolves to {s}", .{ fname, v.spec.hash[0..6], found.spec.hash[0..6] });
    rig.covered[vi] = true;
    rig.cases += 1;
    var lb: [64]u8 = undefined;
    const name = std.fmt.bufPrint(&lb, "{s}[{s}]", .{ fname, v.spec.hash[0..6] }) catch "variant";
    for (rig.part) |b| try rig.clear(b);
    for (rig.out) |b| try b.fill8(0xff, rig.s.handle);
    const t = rig.replay();
    const g = H / HK;
    if (is_shared) {
        try t.replayShared(rig.q.ptr, rig.scratch.origin, offs, plan, rig.part[0].parts(), attn, g, 0.0625);
        try rig.nat.shared(rig.s, rig.q.ptr, rig.scratch.origin, offs, plan, rig.part[1].parts());
        try rig.partials(name, sg.slots, plan.width);
    } else if (is_tail) {
        try t.replayTail(rig.q.ptr, rig.kn.ptr, rig.vn.ptr, rig.scratch.origin, offs, plan, rig.part[0].parts(), attn, g, 0.0625);
        try rig.nat.tail(rig.s, rig.q.ptr, rig.kn.ptr, rig.vn.ptr, rig.scratch.origin, offs, plan, rig.part[1].parts());
        try rig.partials(name, sg.slots, plan.width);
    } else {
        // merge both from the same partials: the native shared and tail kernels' (or the replayed ones) in place
        const aligned = sg.r.attn;
        try rig.nat.shared(rig.s, rig.q.ptr, rig.scratch.origin, offs, aligned, rig.part[0].parts());
        try rig.nat.tail(rig.s, rig.q.ptr, rig.kn.ptr, rig.vn.ptr, rig.scratch.origin, offs, aligned, rig.part[0].parts());
        try t.replayMerge(plan, rig.part[0].parts(), rig.out[0].ptr, attn, g);
        try rig.nat.merge(rig.s, plan, rig.part[0].parts(), rig.out[1].ptr);
        try rig.same("OUT", name, rig.out[0], rig.out[1], plan.width * H * D * 2);
    }
}

// Microseconds one attention call takes back to back, replay against native (information, not a gate).
fn timing(rig: *Rig, specs: []const Spec) !void {
    var par: [16][128]i32 = undefined;
    try rig.load(specs, 0);
    try inputs(rig, false);
    const sg = try stage(rig, specs, &par);
    try rig.s.synchronize();
    var start = try cuda.Event.init(rig.gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(rig.gpu.d, true);
    defer end.deinit();
    const reps = 100;
    const offs = sg.r.offsets(3, 0);
    var us: [2]f64 = undefined;
    for ([_]bool{ false, true }, 0..) |use_native, i| {
        var t = rig.replay();
        if (use_native) t.tree = rig.nat;
        for (0..reps + 10) |n| {
            if (n == 10) try start.record(rig.s);
            try t.attention(rig.q.ptr, rig.kn.ptr, rig.vn.ptr, rig.scratch.origin, offs, sg.r.attn, rig.part[i].parts(), rig.out[i].ptr, attn);
        }
        try end.record(rig.s);
        try end.synchronize();
        us[i] = @as(f64, try cuda.Event.elapsedMs(start, end)) * 1000 / reps;
    }
    std.debug.print("  p {d:>5} rows {d:>3}: replay {d:>7.1} us, native {d:>7.1} us\n", .{ specs[0].p, specs[0].w, us[0], us[1] });
}

/// The engine objects a test rig stands on: stream, kernels (Triton set optional), scratch, streams' caches, the rig.
pub const Env = struct {
    gpa: std.mem.Allocator,
    stream: cuda.Stream,
    k: kern.Kernels,
    scratch: st.Scratch,
    seqs: []st.Seq,
    made: usize,
    rig: Rig,

    pub fn init(env: *Env, gpu: check.Gpu, kernels: ?[]const u8, cap: usize, streams: usize) !void {
        const gpa = gpu.gpa;
        env.gpa = gpa;
        env.stream = try cuda.Stream.init(gpu.d, true);
        errdefer env.stream.deinit();
        env.k = try kern.Kernels.load(gpa, gpu.io, gpu.ctx, kernels, geometry.linear_v_heads);
        errdefer env.k.deinit();
        env.scratch = try st.Scratch.init(gpa, gpu.d, geometry, 256, cap);
        errdefer env.scratch.deinit();
        const o: kern.Ops = .{ .k = &env.k, .s = env.stream };
        env.seqs = try gpa.alloc(st.Seq, streams);
        errdefer gpa.free(env.seqs);
        env.made = 0;
        errdefer for (env.seqs[0..env.made]) |*sq| sq.deinit();
        for (env.seqs) |*sq| {
            sq.* = try st.Seq.init(gpa, o, geometry, cap);
            env.made += 1;
        }
        const part_elems = env.scratch.part_chunks * st.round_rows * H;
        const variants = if (env.k.triton) |t| t.variants.len else 0;
        env.rig = .{
            .gpu = gpu,
            .s = env.stream,
            .k = &env.k,
            .scratch = &env.scratch,
            .seqs = env.seqs,
            .home = undefined,
            .cap = cap,
            .pool = undefined,
            .q = try cuda.DeviceBuffer.alloc(gpu.d, 16),
            .kn = try cuda.DeviceBuffer.alloc(gpu.d, 16),
            .vn = try cuda.DeviceBuffer.alloc(gpu.d, 16),
            .out = .{ try cuda.DeviceBuffer.alloc(gpu.d, st.round_rows * H * D * 2), try cuda.DeviceBuffer.alloc(gpu.d, st.round_rows * H * D * 2) },
            .part = undefined,
            .nat = try native.Native.load(gpu.d, try gpu.ctx.capability()),
            .prng = std.Random.DefaultPrng.init(0x7ee_a77e),
            .covered = try gpa.alloc(bool, variants),
        };
        const rig = &env.rig;
        @memset(rig.covered, false);
        for (env.seqs, rig.home[0..env.seqs.len]) |sq, *h| h.* = .{ sq.keys[3], sq.values[3] };
        for (&rig.part) |*b| b.* = .{ .o = try cuda.DeviceBuffer.alloc(gpu.d, part_elems * D * 4), .m = try cuda.DeviceBuffer.alloc(gpu.d, part_elems * 4), .l = try cuda.DeviceBuffer.alloc(gpu.d, part_elems * 4) };
        for (&rig.pool, 0..) |*pr, i| for (pr) |*b| {
            b.* = try rig.fresh(cap * kv_row + 16 * 4096, i == 1, 1.0);
        };
        try env.stream.synchronize();
    }

    pub fn deinit(env: *Env) void {
        env.gpa.free(env.rig.covered);
        env.rig.nat.deinit();
        for (env.seqs[0..env.made]) |*sq| sq.deinit();
        env.gpa.free(env.seqs);
        env.scratch.deinit();
        env.k.deinit();
        env.stream.deinit();
    }
};

pub fn run(gpu: check.Gpu, kernels: []const u8, rounds: usize) !void {
    var env: Env = undefined;
    try env.init(gpu, kernels, capacity, st.max_streams);
    defer env.deinit();
    const rig = &env.rig;
    const k = &env.k;
    grid(rig);
    std.debug.print("grid: {d} cases, {d} comparisons, {d} differ, {d} skipped (no captured variant)\n", .{ rig.cases, rig.comparisons, rig.bad, rig.skipped });
    const before = rig.bad;
    lanes(rig, rounds);
    std.debug.print("lanes: {d} cases so far, {d} differ\n", .{ rig.cases, rig.bad - before });
    for (0..k.triton.?.variants.len) |vi| variant(rig, vi) catch |e| {
        std.debug.print("  variant {d} failed: {t}\n", .{ vi, e });
        rig.bad += 1;
    };
    for ([_]Spec{ .{ .p = 4096, .w = 1 }, .{ .p = 4096, .w = 16 }, .{ .p = 8191, .w = 1 }, .{ .p = 8191, .w = 16 }, .{ .p = 8191, .w = 64 } }) |sp| try timing(rig, &.{sp});
    var missing: usize = 0;
    for (k.triton.?.variants, rig.covered) |v, c| {
        const f = v.spec.@"fn";
        if (!c and (std.mem.eql(u8, f, "_shared") or std.mem.eql(u8, f, "_tail") or std.mem.eql(u8, f, "_merge"))) missing += 1;
    }
    std.debug.print("variants: {d} of the captured shared/tail/merge entries not exercised\n", .{missing});
    try check.expect(rig.bad == 0, "{d} of {d} comparisons differ", .{ rig.bad, rig.comparisons });
    try check.expect(missing == 0, "{d} captured variants not exercised", .{missing});
    check.pass("qwen-attention: native _shared, _tail and _merge byte-equal to the captured Triton launches over {d} cases, {d} comparisons, {d} MiB compared, {d} Mwords written ({d} skipped)", .{ rig.cases, rig.comparisons, rig.bytes >> 20, rig.written >> 20, rig.skipped });
}
