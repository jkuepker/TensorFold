//! The gated delta rule kernels qwen3_5 launches (gdn.cu tree and replay, gdn_prefill.cu chain) against float64.

const std = @import("std");
const cuda = @import("cuda");
const qwen27 = @import("qwen27");
const check = @import("check.zig");
const f64t = @import("qwen_f64.zig");
const att = @import("qwen_attention_tests.zig");
const ref = @import("qwen_gdn_f64_ref.zig");

const kern = qwen27.kernels;
const Pair = f64t.Pair;
const hk = ref.hk;
const hv = ref.hv;
const dk = ref.dk;
const dv = ref.dv;
const heads: kern.Heads = .{ .hk = hk, .hv = hv, .dv = dv };

const Data = struct {
    q: []u16,
    k: []u16,
    v: []u16,
    g: []f32,
    beta: []f32,

    // gdn_pre's outputs in kind: unit k, q by 1/sqrt(dk), v by head, g from a decay rate per head, beta a sigmoid
    fn make(gpa: std.mem.Allocator, rng: std.Random, rows: usize) !Data {
        var d: Data = .{ .q = try gpa.alloc(u16, rows * hk * dk), .k = try gpa.alloc(u16, rows * hk * dk), .v = try gpa.alloc(u16, rows * hv * dv), .g = try gpa.alloc(f32, rows * hv), .beta = try gpa.alloc(f32, rows * hv) };
        const rates = [_]f64{ 0.001, 0.01, 0.05, 0.3, 1.2 };
        const vs = [_]f32{ 0.3, 1, 3 };
        for (0..rows) |r| {
            for (0..hk) |h| for ([_][]u16{ d.k, d.q }, [_]f64{ 1, 1.0 / @sqrt(128.0) }) |dst, scale| {
                var c: [dk]f32 = undefined;
                var ss: f64 = 0;
                for (&c) |*x| {
                    x.* = rng.floatNorm(f32);
                    ss += @as(f64, x.*) * x.*;
                }
                for (dst[(r * hk + h) * dk ..][0..dk], c) |*o, x| o.* = f64t.bf16Of(@floatCast(x / @sqrt(ss) * scale));
            };
            for (0..hv) |h| {
                for (d.v[(r * hv + h) * dv ..][0..dv]) |*o| o.* = f64t.bf16Of(rng.floatNorm(f32) * vs[h % vs.len]);
                d.g[r * hv + h] = @floatCast(@exp(-rates[h % rates.len] * @exp(rng.floatNorm(f64) * 0.4)));
                d.beta[r * hv + h] = @floatCast(1 / (1 + @exp(-rng.floatNorm(f64) * 2)));
            }
        }
        return d;
    }

    fn free(d: Data, gpa: std.mem.Allocator) void {
        gpa.free(d.q);
        gpa.free(d.k);
        gpa.free(d.v);
        gpa.free(d.g);
        gpa.free(d.beta);
    }

    fn inputs(d: Data) ref.Inputs {
        return .{ .q = d.q, .k = d.k, .v = d.v, .g = d.g, .beta = d.beta };
    }
};

const Dev = struct {
    q: cuda.DeviceBuffer,
    k: cuda.DeviceBuffer,
    v: cuda.DeviceBuffer,
    g: cuda.DeviceBuffer,
    beta: cuda.DeviceBuffer,

    fn up(d: *const cuda.Driver, x: Data) !Dev {
        return .{ .q = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(x.q)), .k = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(x.k)), .v = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(x.v)), .g = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(x.g)), .beta = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(x.beta)) };
    }

    fn free(x: *Dev) void {
        for ([_]*cuda.DeviceBuffer{ &x.q, &x.k, &x.v, &x.g, &x.beta }) |b| b.free();
    }
};

/// The comparison of one window: output rows y and, where `last` is given, the state after its final node.
const Cmp = struct {
    y: ?*Pair = null,
    y_raw: []const u8 = &.{},
    first: usize = 0,
    last: ?*Pair = null,
    last_raw: []const u8 = &.{},
    len: usize = 0,

    pub fn node(c: *const Cmp, head: usize, j: usize, out: *const ref.NodeOut, state: []const f64) void {
        if (c.y) |p| for (0..dv) |row| p.add(row, f64t.bfAt(c.y_raw, ((c.first + j) * hv + head) * dv + row), null, out.y[row], out.y_bound[row], out.y_floor[row]);
        if (c.last) |p| if (j + 1 == c.len) for (0..dv) |row| for (0..dk) |i| {
            p.add(i, f64t.f32At(c.last_raw, (head * dv + row) * dk + i), null, state[row * dk + i], out.s_bound[row], out.s_floor[row]);
        };
    }
};

const Ctx = struct {
    gpu: check.Gpu,
    o: kern.Ops,
    rng: std.Random,
    chain_y: Pair,
    tree_y: Pair,
    prompt_y: Pair,
    prompt_last: Pair,
    same_prompt: Pair, // the rows both paths run: prompt chunk and decode chain
    same_decode: Pair,
    replay_state: Pair,
    cases: usize = 0,
    nodes: usize = 0,
    variants: [6]bool = @splat(false),
    unchanged_bad: usize = 0,
    words_differ: usize = 0, // bf16 words of y that differ between the prompt chain and the decode chain on identical rows
    words_same_rows: usize = 0,

    fn state(c: *Ctx, zero: bool) ![]f32 {
        const s = try c.gpu.gpa.alloc(f32, ref.state_len);
        for (s) |*x| x.* = if (zero) 0 else c.rng.floatNorm(f32) * 0.05;
        return s;
    }

    fn fetch(c: *Ctx, b: cuda.DeviceBuffer) ![]u8 {
        try c.o.s.synchronize();
        return check.download(c.gpu, b);
    }
};

fn fresh(c: *Ctx, bytes: usize) !cuda.DeviceBuffer {
    const b = try cuda.DeviceBuffer.alloc(c.gpu.d, bytes);
    try b.fill8(0xff, c.o.s.handle);
    return b;
}

const Kind = enum { chain, bushy, comb, wide };
const Win = struct { w: usize, kind: Kind };

// Parents of a window: a chain, short random branches, or a quarter-length spine with leaves on random spine nodes.
fn parentsFor(rng: std.Random, kind: Kind, out: []i32) void {
    switch (kind) {
        .chain => att.parentsOf(rng, .chain, out),
        .bushy => att.parentsOf(rng, .bushy, out),
        .wide => for (out, 0..) |*x, i| {
            x.* = if (i == 0) -1 else @intCast(i - 1 - rng.uintLessThan(usize, @min(i, 40)));
        },
        .comb => for (out, 0..) |*x, i| {
            const spine = @max(out.len / 4, 1);
            x.* = if (i == 0) -1 else if (i < spine) @as(i32, @intCast(i)) - 1 else @intCast(rng.uintLessThan(usize, spine));
        },
    }
}

// The schedule-slot classes the kernel's variants cover on this GPU: chain, then up to 2, 4, 8, 16 and more slots.
fn bucket(slots: usize) usize {
    return if (slots == 0) 0 else if (slots <= 2) 1 else if (slots <= 4) 2 else if (slots <= 8) 3 else if (slots <= 16) 4 else 5;
}

// One round of verify windows (one stream, or several through a state table) on `data` and `states`.
fn treeRound(c: *Ctx, data: Data, states: []const []f32, wins: []const Win, force: usize, into: ?*Pair) ![]u8 {
    const gpa = c.gpu.gpa;
    const S = wins.len;
    var rows: usize = 0;
    var most: usize = 0;
    var par: [16][128]i32 = undefined;
    var plan: std.ArrayList(i32) = .empty;
    defer plan.deinit(gpa);
    var starts: [17]i32 = undefined;
    var slots: usize = 0;
    for (wins, 0..) |wn, i| {
        parentsFor(c.rng, wn.kind, par[i][0..wn.w]);
        const sc = try qwen27.tree.schedule(gpa, par[i][0..wn.w]);
        defer gpa.free(sc.entries);
        for (sc.entries, 0..) |x, j| try plan.append(gpa, if (j % 3 == 0) x + @as(i32, @intCast(rows)) else x);
        slots = @max(slots, sc.slots);
        starts[i] = @intCast(rows);
        rows += wn.w;
        most = @max(most, wn.w);
    }
    starts[S] = @intCast(rows);
    slots = if (slots == 0) 0 else @max(slots, force); // a larger capacity than the plan needs runs the wider variants
    var dev = try Dev.up(c.gpu.d, data);
    defer dev.free();
    var dstates: [16]cuda.DeviceBuffer = undefined;
    var ptrs: [16]u64 = undefined;
    for (0..S) |i| {
        dstates[i] = try cuda.DeviceBuffer.fromHost(c.gpu.d, std.mem.sliceAsBytes(states[i]));
        ptrs[i] = dstates[i].ptr;
    }
    defer for (dstates[0..S]) |*b| b.free();
    var table = try cuda.DeviceBuffer.fromHost(c.gpu.d, std.mem.sliceAsBytes(ptrs[0..S]));
    defer table.free();
    var dstarts = try cuda.DeviceBuffer.fromHost(c.gpu.d, std.mem.sliceAsBytes(starts[0 .. S + 1]));
    defer dstarts.free();
    var dplan = try cuda.DeviceBuffer.fromHost(c.gpu.d, std.mem.sliceAsBytes(plan.items));
    defer dplan.free();
    var y = try fresh(c, rows * hv * dv * 2);
    defer y.free();
    const multi = S > 1;
    try c.o.gdnTree(dev.q.ptr, dev.k.ptr, dev.v.ptr, dev.g.ptr, dev.beta.ptr, if (multi) 0 else dstates[0].ptr, if (multi) table.ptr else 0, if (multi) dstarts.ptr else 0, dplan.ptr, y.ptr, rows, S, slots, most, heads);
    const raw = try c.fetch(y);
    errdefer gpa.free(raw);
    const in = data.inputs();
    const pair = into orelse if (slots == 0) &c.chain_y else &c.tree_y;
    for (wins, 0..) |wn, i| {
        const cmp: Cmp = .{ .y = pair, .y_raw = raw, .first = @intCast(starts[i]) };
        for (0..hv) |h| try ref.tree(gpa, &cmp, &in, h, @intCast(starts[i]), par[i][0..wn.w], states[i][h * dv * dk ..][0 .. dv * dk]);
        c.nodes += wn.w;
    }
    c.variants[bucket(slots)] = true;
    c.cases += 1;
    return raw;
}

fn treeCase(c: *Ctx, wins: []const Win, forces: []const usize) !void {
    const gpa = c.gpu.gpa;
    var rows: usize = 0;
    for (wins) |wn| rows += wn.w;
    const data = try Data.make(gpa, c.rng, rows);
    defer data.free(gpa);
    var states: [16][]f32 = undefined;
    for (wins, 0..) |_, i| states[i] = try c.state(i % 3 == 0);
    defer for (states[0..wins.len]) |s| gpa.free(s);
    for (forces) |f| gpa.free(try treeRound(c, data, states[0..wins.len], wins, f, null));
}

// The same rows as a prompt chunk (chain kernel) and as a decode chain (tree kernel), each against float64.
fn promptCase(c: *Ctx, rows: usize, zero: bool, decode_too: bool) !void {
    const gpa = c.gpu.gpa;
    const data = try Data.make(gpa, c.rng, rows);
    defer data.free(gpa);
    const st = try c.state(zero);
    defer gpa.free(st);
    var dev = try Dev.up(c.gpu.d, data);
    defer dev.free();
    var dstate = try cuda.DeviceBuffer.fromHost(c.gpu.d, std.mem.sliceAsBytes(st));
    defer dstate.free();
    var last = try fresh(c, ref.state_len * 4);
    defer last.free();
    var y = try fresh(c, rows * hv * dv * 2);
    defer y.free();
    try c.o.gdnPrompt(dev.q.ptr, dev.k.ptr, dev.v.ptr, dev.g.ptr, dev.beta.ptr, dstate.ptr, last.ptr, y.ptr, rows, heads);
    const raw = try c.fetch(y);
    defer gpa.free(raw);
    const lraw = try c.fetch(last);
    defer gpa.free(lraw);
    const nodes = try gpa.alloc(usize, rows);
    defer gpa.free(nodes);
    for (nodes, 0..) |*n, i| n.* = i;
    const in = data.inputs();
    const cmp: Cmp = .{ .y = if (decode_too) &c.same_prompt else &c.prompt_y, .y_raw = raw, .last = &c.prompt_last, .last_raw = lraw, .len = rows };
    for (0..hv) |h| try ref.chain(gpa, &cmp, &in, h, nodes, st[h * dv * dk ..][0 .. dv * dk]);
    c.nodes += rows;
    c.cases += 1;
    if (!decode_too) return;
    const dec = try treeRound(c, data, &.{st}, &.{.{ .w = rows, .kind = .chain }}, 0, &c.same_decode);
    defer gpa.free(dec);
    for (0..rows * hv * dv) |i| c.words_differ += @intFromBool(f64t.bfAt(raw, i) != f64t.bfAt(dec, i));
    c.words_same_rows += rows * hv * dv;
}

// Committed rows stepped into each stream's states in place: the accepted path of each stream, two layers.
fn replayCase(c: *Ctx, streams: usize, max_rows: usize) !void {
    const gpa = c.gpu.gpa;
    const layers = 2;
    const rows = 40;
    var data: [layers]Data = undefined;
    var dev: [layers]Dev = undefined;
    for (&data, &dev) |*d, *v| {
        d.* = try Data.make(gpa, c.rng, rows);
        v.* = try Dev.up(c.gpu.d, d.*);
    }
    defer for (&data, &dev) |d, *v| {
        d.free(gpa);
        v.free();
    };
    var table: [4 * layers + 16 * layers]u64 = undefined;
    for (dev, 0..) |v, l| table[4 * l ..][0..4].* = .{ v.k.ptr, v.v.ptr, v.g.ptr, v.beta.ptr };
    var host: [16][layers][]f32 = undefined;
    var dst: [16][layers]cuda.DeviceBuffer = undefined;
    var made: usize = 0;
    defer for (0..made) |i| for (0..layers) |l| {
        gpa.free(host[i][l]);
        dst[i][l].free();
    };
    var path: [16][64]i32 = undefined;
    var count: [16]i32 = undefined;
    for (0..streams) |s| {
        count[s] = if (s == 1 and streams > 2) 0 else @intCast(1 + c.rng.uintLessThan(usize, max_rows));
        var at: i32 = -1;
        for (path[s][0..@intCast(count[s])]) |*r| {
            at += 1 + @as(i32, @intCast(c.rng.uintLessThan(usize, 2)));
            r.* = @min(at, rows - 1);
        }
        for (0..layers) |l| {
            host[s][l] = try c.state(s % 2 == 0);
            dst[s][l] = try cuda.DeviceBuffer.fromHost(c.gpu.d, std.mem.sliceAsBytes(host[s][l]));
            table[4 * layers + s * layers + l] = dst[s][l].ptr;
        }
        made += 1;
    }
    var dtable = try cuda.DeviceBuffer.fromHost(c.gpu.d, std.mem.sliceAsBytes(table[0 .. 4 * layers + streams * layers]));
    defer dtable.free();
    var drows = try cuda.DeviceBuffer.fromHost(c.gpu.d, std.mem.sliceAsBytes(path[0..streams]));
    defer drows.free();
    var dcount = try cuda.DeviceBuffer.fromHost(c.gpu.d, std.mem.sliceAsBytes(count[0..streams]));
    defer dcount.free();
    try c.o.gdnReplay(dtable.ptr, layers, streams, drows.ptr, 64, dcount.ptr, 1, heads);
    for (0..streams) |s| for (0..layers) |l| {
        const raw = try c.fetch(dst[s][l]);
        defer gpa.free(raw);
        const n: usize = @intCast(count[s]);
        if (n == 0) {
            if (!std.mem.eql(u8, raw, std.mem.sliceAsBytes(host[s][l]))) c.unchanged_bad += 1;
            continue;
        }
        const nodes = try gpa.alloc(usize, n);
        defer gpa.free(nodes);
        for (nodes, path[s][0..n]) |*x, r| x.* = @intCast(r);
        const in = data[l].inputs();
        const cmp: Cmp = .{ .last = &c.replay_state, .last_raw = raw, .len = n };
        for (0..hv) |h| try ref.chain(gpa, &cmp, &in, h, nodes, host[s][l][h * dv * dk ..][0 .. dv * dk]);
        c.nodes += n;
    };
    c.cases += 1;
}

// slot capacities beyond a plan's need, one for each wider variant (the natural count first)
const widen = [_]usize{ 0, 3, 5, 9, 17 };

pub fn run(gpu: check.Gpu) !void {
    const gpa = gpu.gpa;
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var k = try kern.Kernels.load(gpa, gpu.io, gpu.ctx, null, hv);
    defer k.deinit();
    var prng = std.Random.DefaultPrng.init(0x9d17a);
    var c: Ctx = .{
        .gpu = gpu,
        .o = .{ .k = &k, .s = stream },
        .rng = prng.random(),
        .chain_y = try Pair.init(gpa, "gdn.decode_chain_y", dv),
        .tree_y = try Pair.init(gpa, "gdn.tree_y", dv),
        .prompt_y = try Pair.init(gpa, "gdn.prompt_chain_y", dv),
        .prompt_last = try Pair.init(gpa, "gdn.prompt_chain_state", dk),
        .same_prompt = try Pair.init(gpa, "gdn.identical_rows.prompt_chain_y", dv),
        .same_decode = try Pair.init(gpa, "gdn.identical_rows.decode_chain_y", dv),
        .replay_state = try Pair.init(gpa, "gdn.replay_state", dk),
    };
    for ([_]usize{ 1, 2, 3, 4, 8, 16, 17, 32, 64, 128 }) |w| {
        try treeCase(&c, &.{.{ .w = w, .kind = .chain }}, &.{0});
        if (w >= 3) try treeCase(&c, &.{.{ .w = w, .kind = .bushy }}, if (w >= 16) &widen else &.{0});
        if (w >= 16) try treeCase(&c, &.{.{ .w = w, .kind = .comb }}, &widen);
        if (w >= 16) try treeCase(&c, &.{.{ .w = w, .kind = .wide }}, &widen);
    }
    for ([_]usize{ 2, 5, 16 }) |streams| {
        var wins: [16]Win = undefined;
        for (wins[0..streams], 0..) |*wn, i| wn.* = .{ .w = 1 + c.rng.uintLessThan(usize, 16), .kind = if (i % 3 == 0) .chain else if (i % 3 == 1) .bushy else .comb };
        try treeCase(&c, wins[0..streams], &widen);
        for (wins[0..streams]) |*wn| wn.kind = .chain;
        try treeCase(&c, wins[0..streams], &.{0});
    }
    std.debug.print("verify windows: {d} cases, {d} nodes\n", .{ c.cases, c.nodes });
    for ([_]usize{ 1, 16, 64, 128 }) |rows| try promptCase(&c, rows, rows % 2 == 0, true);
    for ([_]usize{ 300, 1024, 2048, 4096 }) |rows| try promptCase(&c, rows, false, false);
    for ([_]usize{ 1, 3, 16 }) |streams| try replayCase(&c, streams, 16);
    var seen: usize = 0;
    for (c.variants) |v| seen += @intFromBool(v);
    std.debug.print("{d} cases, {d} nodes, {d} of 6 schedule-slot classes\n", .{ c.cases, c.nodes, seen });
    try check.expect(seen == 6, "only {d} of the 6 tree kernel variants ran", .{seen});
    var failed = false;
    try f64t.gate(&failed, c.chain_y.finish("fp32 steps 32 x 2^-24 of the row norm, decayed by g down the path, through |q|, + bf16 half-ulp", true));
    try f64t.gate(&failed, c.tree_y.finish("fp32 steps 32 x 2^-24 of the row norm, decayed by g down the tree path, through |q|, + bf16 half-ulp", true));
    try f64t.gate(&failed, c.prompt_y.finish("the same bound on the same rows as the decode chain", true));
    try f64t.gate(&failed, c.same_prompt.finish("identical rows through the prompt chain kernel", true));
    try f64t.gate(&failed, c.same_decode.finish("identical rows through the decode chain (tree kernel)", true));
    try f64t.gate(&failed, c.prompt_last.finish("fp32 steps 32 x 2^-24 of the row norm, decayed by g (state element)", true));
    try f64t.gate(&failed, c.replay_state.finish("fp32 steps 32 x 2^-24 of the row norm, decayed by g (state element)", true));
    std.debug.print("identical rows: the prompt and decode chains' y differ in {d} of {d} bf16 words\n", .{ c.words_differ, c.words_same_rows });
    try check.expect(!failed, "qwen-gdn-f64: an output left its float64 bound", .{});
    try check.expect(c.unchanged_bad == 0, "{d} empty replays changed a state", .{c.unchanged_bad});
    check.pass("qwen-gdn-f64: gdn tree, prompt chain and replay within their float64 bounds over {d} cases, {d} nodes", .{ c.cases, c.nodes });
}
