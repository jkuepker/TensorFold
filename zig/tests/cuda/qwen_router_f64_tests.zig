//! Qwen3.6's router against host float64: logits, top-k ranking and weights, the shared gate, the slot-order combine.
const std = @import("std");
const cuda = @import("cuda");
const router = @import("qwen35_cuda").router;
const check = @import("check.zig");
const st = @import("qwen_f64.zig");
const rep = @import("qwen_moe_f64_report.zig");

const Gpu = check.Gpu;

// The logits' bound: one fp32 rounding per k16 step, 128 steps over the 2048 inputs, of sum |x w|.
const logit_bound = 0x1p-16;
// A softmax weight is a bf16 of exp / sum: ex2.approx and div.full add 2^-20 relative.
const weight_eps = st.eps_bf16 + 0x1p-20;
// The combine's nine fp32 fused multiply-adds, each within 2^-24 of the running sum of |w y|.
const combine_bound = 0x1p-20;

const Out = enum { logits, weights, combine_f32, combine_bf16 };

const Rig = struct {
    gpu: Gpu,
    s: cuda.Stream,
    native: router.Native,
    replay: ?router.Replay,
    sh: router.Shape,
    checks: usize = 0,
    ties: usize = 0, // rows whose picks differ from the strict float64 ranking, each inside the logits' bounds
    controls: u8 = 0, // the negative controls already shown, a bit each
    worst: [4]rep.Out = undefined, // by Out, every case folded in, set once the shape is known

    fn up(r: *Rig, bytes: []const u8) !cuda.DeviceBuffer {
        return cuda.DeviceBuffer.fromHost(r.gpu.d, bytes);
    }

    // An output the kernel must fill whole: 0xff bytes first, so an unwritten word is NaN.
    fn out(r: *Rig, n: usize) !cuda.DeviceBuffer {
        const b = try cuda.DeviceBuffer.alloc(r.gpu.d, n);
        try b.fill8(0xff, r.s.handle);
        return b;
    }

    fn read(r: *Rig, a: std.mem.Allocator, b: cuda.DeviceBuffer) ![]u8 {
        try r.s.synchronize();
        const host = try a.alloc(u8, b.len);
        try b.download(0, host);
        return host;
    }

    // A negative control is shown once a run: the first time `bit` asks.
    fn first(r: *Rig, bit: u3) bool {
        const seen = r.controls & (@as(u8, 1) << bit) != 0;
        r.controls |= @as(u8, 1) << bit;
        return !seen;
    }

    fn note(r: *Rig, which: Out, c: *const st.Phased) void {
        r.worst[@intFromEnum(which)].fold(c);
    }

    fn f64s(r: *Rig, a: std.mem.Allocator, b: cuda.DeviceBuffer, kind: st.Kind) ![]f64 {
        const host = try r.read(a, b);
        return st.decode(a, host, kind, host.len / (if (kind == .f32) @as(usize, 4) else 2));
    }
};

fn dots(a: std.mem.Allocator, x: []const u16, w: []const u16, m: usize, ne: usize, d: usize, want: []f64, tol: []f64) !void {
    const xf = try a.alloc(f64, x.len);
    defer a.free(xf);
    const wf = try a.alloc(f64, w.len);
    defer a.free(wf);
    for (xf, x) |*o, v| o.* = st.bf(v);
    for (wf, w) |*o, v| o.* = st.bf(v);
    for (0..m) |i| {
        for (0..ne) |e| {
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (xf[i * d ..][0..d], wf[e * d ..][0..d]) |p, q| {
                sum += p * q;
                abs += @abs(p * q);
            }
            want[i * ne + e] = sum;
            tol[i * ne + e] = logit_bound * abs + 1e-30;
        }
    }
}

fn tag(buf: []u8, name: []const u8, what: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "{s} {s}", .{ name, what }) catch name;
}

const Rank = struct { valid: bool, strict: bool, slot: usize = 0, other: usize = 0 };

// Picks `p` vs float64 logits `l` (`t`: the fp32 logits' errors): a top-k up to them, strict or one swap off.
fn rank(l: []const f64, t: []const f64, p: []const i32, top_k: usize) Rank {
    const ne = l.len;
    var in: [257]bool = @splat(false);
    for (p[0..top_k]) |id| {
        if (id < 0 or id >= ne or in[@intCast(id)]) return .{ .valid = false, .strict = false };
        in[@intCast(id)] = true;
    }
    var tolerant = true;
    for (0..top_k - 1) |k| {
        const a: usize = @intCast(p[k]);
        const b: usize = @intCast(p[k + 1]);
        if (!(l[a] + t[a] + t[b] >= l[b])) tolerant = false;
    }
    const last: usize = @intCast(p[top_k - 1]);
    for (0..ne) |e| {
        if (!in[e] and !(l[e] <= l[last] + t[e] + t[last])) tolerant = false;
    }
    // the strict ranking, the lower id on equal logits; one past the picks for a boundary swap
    var q: [17]usize = undefined;
    var taken: [257]bool = @splat(false);
    for (0..top_k + 1) |k| {
        var best: usize = ne;
        for (0..ne) |e| {
            if (!taken[e] and (best == ne or l[e] > l[best])) best = e;
        }
        taken[best] = true;
        q[k] = best;
    }
    const at = for (0..top_k) |k| {
        if (p[k] != q[k]) break k;
    } else return .{ .valid = tolerant, .strict = true };
    const swap = at + 1 < top_k and p[at] == q[at + 1] and p[at + 1] == q[at] and for (at + 2..top_k) |k| {
        if (p[k] != q[k]) break false;
    } else true;
    const boundary = at == top_k - 1 and p[at] == q[top_k];
    return .{ .valid = tolerant and (swap or boundary), .strict = false, .slot = at, .other = q[at] };
}

// Picks and weights of `select` vs float64 logits `lref`; `lnat` is what select read, `native` counts near-ties.
fn pickAndWeigh(r: *Rig, a: std.mem.Allocator, name: []const u8, lnat: []const f64, lref: []const f64, native: bool, picks: []const i32, wts: []const f64) !rep.Run {
    const sh = r.sh;
    const ne = sh.experts + 1;
    const slots = sh.top_k + 1;
    const m = lref.len / ne;
    const want = try a.alloc(f64, m * slots);
    const tol = try a.alloc(f64, m * slots);
    for (0..m) |row| {
        const l = lref[row * ne ..][0..ne];
        const p = picks[row * slots ..][0..slots];
        const derr = try a.alloc(f64, sh.experts);
        for (derr, 0..) |*v, e| v.* = @abs(lnat[row * ne + e] - l[e]);
        const rk = rank(l[0..sh.experts], derr, p, sh.top_k);
        try check.expect(rk.valid and p[sh.top_k] == @as(i32, @intCast(sh.experts)), "{s}: row {d} picks are a top-{d} of the float64 logits", .{ name, row, sh.top_k });
        if (!rk.strict and native) {
            r.ties += 1;
            const a_id: usize = @intCast(p[rk.slot]);
            std.debug.print("  {s}: row {d} slot {d}: picked {d}, float64 ranks {d} first by {e:.2} (logit errors {e:.2} + {e:.2})\n", .{ name, row, rk.slot, a_id, rk.other, l[rk.other] - l[a_id], derr[a_id], derr[rk.other] });
        }
        var top: f64 = -std.math.inf(f64);
        var dl: f64 = 0;
        for (p[0..sh.top_k]) |id| {
            top = @max(top, l[@intCast(id)]);
            dl = @max(dl, @abs(lnat[row * ne + @as(usize, @intCast(id))] - l[@intCast(id)]));
        }
        var total: f64 = 0;
        for (p[0..sh.top_k]) |id| total += @exp(l[@intCast(id)] - top);
        for (p[0..sh.top_k], 0..) |id, k| {
            const w = @exp(l[@intCast(id)] - top) / total;
            want[row * slots + k] = w;
            tol[row * slots + k] = weight_eps * w + 2 * w * dl + 1e-30;
        }
        // the shared slot: sigmoid of the gate logit, rounded to bf16 first (an interval: |g| can be 100)
        const g = l[sh.experts];
        const gn = lnat[row * ne + sh.experts];
        const delta = @abs(gn - g) + st.eps_bf16 * @abs(gn);
        const sg = st.sigmoid(g);
        want[row * slots + sh.top_k] = sg;
        tol[row * slots + sh.top_k] = @max(sg - st.sigmoid(g - delta), st.sigmoid(g + delta) - sg) + weight_eps * st.sigmoid(g + delta) + 1e-30;
    }
    if (native and r.first(0)) try rep.control(a, "router select weights", wts, want, tol, slots);
    r.checks += 1;
    return .{ .got = wts, .want = want, .tol = tol };
}

// The ranking check must refuse two picks swapped where the float64 logits are far apart.
fn swapControl(r: *Rig, name: []const u8, lnat: []const f64, lref: []const f64, picks: []const i32) !void {
    const sh = r.sh;
    const ne = sh.experts + 1;
    const slots = sh.top_k + 1;
    for (0..lref.len / ne) |row| {
        const l = lref[row * ne ..][0..ne];
        const a: usize = @intCast(picks[row * slots]);
        const b: usize = @intCast(picks[row * slots + 1]);
        if (!(l[a] - l[b] > 4 * (@abs(lnat[row * ne + a] - l[a]) + @abs(lnat[row * ne + b] - l[b])) + 1e-3)) continue;
        var p: [16]i32 = undefined;
        @memcpy(p[0..slots], picks[row * slots ..][0..slots]);
        std.mem.swap(i32, &p[0], &p[1]);
        var derr: [256]f64 = undefined;
        for (derr[0..sh.experts], 0..) |*v, e| v.* = @abs(lnat[row * ne + e] - l[e]);
        const rk = rank(l[0..sh.experts], derr[0..sh.experts], &p, sh.top_k);
        std.debug.print("CONTROL {s}: row {d} first two picks swapped (float64 gap {d:.3}) -> {s}\n", .{ name, row, l[a] - l[b], if (rk.valid) "ACCEPTED" else "refused" });
        try check.expect(!rk.valid, "{s}: swapped picks are refused", .{name});
        return;
    }
}

// select on the logits at `ldev` by the native kernel and, with a captured set, the Triton launch.
fn select(r: *Rig, a: std.mem.Allocator, name: []const u8, ldev: cuda.DeviceBuffer, lref: []const f64) !void {
    const sh = r.sh;
    const slots = sh.top_k + 1;
    const m = lref.len / (sh.experts + 1);
    const lnat = try r.f64s(a, ldev, .f32);
    var label: [160]u8 = undefined;
    var c = try st.Phased.init(a, tag(&label, name, "select weights"), slots);
    var native: rep.Run = undefined;
    for ([_]bool{ false, true }) |triton| {
        if (triton and r.replay == null) break;
        var pb = try r.out(m * slots * 4);
        defer pb.free();
        var wb = try r.out(m * slots * 4);
        defer wb.free();
        if (triton) try r.replay.?.select(ldev.ptr, pb.ptr, wb.ptr, m, sh) else try r.native.select(ldev.ptr, pb.ptr, wb.ptr, m, sh);
        const praw = try r.read(a, pb);
        const picks = try a.alloc(i32, m * slots);
        for (picks, 0..) |*v, i| v.* = std.mem.bytesToValue(i32, praw[4 * i ..][0..4]);
        var who: [160]u8 = undefined;
        const e = try pickAndWeigh(r, a, tag(&who, name, if (triton) "triton weights" else "native weights"), lnat, lref, !triton, picks, try r.f64s(a, wb, .f32));
        if (triton) {
            try rep.feed(c.at(m), native, e);
        } else {
            native = e;
            if (r.replay == null) try rep.feed(c.at(m), e, null);
            if (r.first(1)) try swapControl(r, "router picks", lnat, lref, picks);
        }
    }
    try rep.show(&c);
    r.note(.weights, &c);
}

// Logits of `x` rows against `w` router rows, then select on them: native and Triton against float64.
fn routerCase(r: *Rig, gpa: std.mem.Allocator, name: []const u8, x: []const u16, w: []const u16) !void {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const sh = r.sh;
    const ne = sh.experts + 1;
    const m = x.len / sh.d;
    const want = try a.alloc(f64, m * ne);
    const tol = try a.alloc(f64, m * ne);
    try dots(a, x, w, m, ne, sh.d, want, tol);
    var xb = try r.up(std.mem.sliceAsBytes(x));
    defer xb.free();
    var wb = try r.up(std.mem.sliceAsBytes(w));
    defer wb.free();
    var ln = try r.out(m * ne * 4);
    defer ln.free();
    try r.native.logits(xb.ptr, wb.ptr, ln.ptr, m, sh);
    const nat: rep.Run = .{ .got = try r.f64s(a, ln, .f32), .want = want, .tol = tol };
    var tri: ?rep.Run = null;
    if (r.replay) |rp| {
        var lt = try r.out(m * ne * 4);
        defer lt.free();
        try rp.logits(xb.ptr, wb.ptr, lt.ptr, m, sh);
        tri = .{ .got = try r.f64s(a, lt, .f32), .want = want, .tol = tol };
    }
    var label: [160]u8 = undefined;
    var c = try st.Phased.init(a, try std.fmt.bufPrint(&label, "{s} logits [{d}, {d}] x [{d}]", .{ name, m, sh.d, ne }), ne);
    try rep.feed(c.at(m), nat, tri);
    try rep.show(&c);
    if (r.first(2)) try rep.control(a, "router logits", nat.got, want, tol, ne);
    r.note(.logits, &c);
    r.checks += 1;
    try select(r, a, name, ln, want);
}

// Selection on given logits (no matmul): float64 logits are the input bytes, so the ranking is exact, ties by lower id.
fn selectCase(r: *Rig, gpa: std.mem.Allocator, name: []const u8, l: []const f32) !void {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var lb = try r.up(std.mem.sliceAsBytes(l));
    defer lb.free();
    const lref = try a.alloc(f64, l.len);
    for (lref, l) |*o, v| o.* = v;
    try select(r, a, name, lb, lref);
}

// The combine of `y` (fp32, or bf16) and weights `w` against float64 sums, native and Triton.
fn combineCase(r: *Rig, gpa: std.mem.Allocator, name: []const u8, y: []const u8, y_bf16: bool, w: []const f32) !void {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const sh = r.sh;
    const slots = sh.top_k + 1;
    const m = w.len / slots;
    const yf = try st.decode(a, y, if (y_bf16) .bf16 else .f32, m * slots * sh.d);
    const want = try a.alloc(f64, m * sh.d);
    const tol = try a.alloc(f64, m * sh.d);
    for (0..m) |row| {
        for (0..sh.d) |c| {
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (0..slots) |k| {
                const t = @as(f64, w[row * slots + k]) * yf[(row * slots + k) * sh.d + c];
                sum += t;
                abs += @abs(t);
            }
            want[row * sh.d + c] = sum;
            tol[row * sh.d + c] = st.eps_bf16 * @abs(sum) + combine_bound * abs + 1e-30;
        }
    }
    var yb = try r.up(y);
    defer yb.free();
    var wb = try r.up(std.mem.sliceAsBytes(w));
    defer wb.free();
    var nat: rep.Run = undefined;
    var tri: ?rep.Run = null;
    for ([_]bool{ false, true }) |triton| {
        if (triton and r.replay == null) break;
        var ob = try r.out(m * sh.d * 2);
        defer ob.free();
        if (triton) try r.replay.?.combine(yb.ptr, y_bf16, wb.ptr, ob.ptr, m, sh) else try r.native.combine(yb.ptr, y_bf16, wb.ptr, ob.ptr, m, sh);
        const res: rep.Run = .{ .got = try r.f64s(a, ob, .bf16), .want = want, .tol = tol };
        if (triton) tri = res else nat = res;
    }
    var label: [160]u8 = undefined;
    var c = try st.Phased.init(a, try std.fmt.bufPrint(&label, "{s} combine {s} y, {d} rows", .{ name, if (y_bf16) "bf16" else "fp32", m }), sh.d);
    try rep.feed(c.at(m), nat, tri);
    try rep.show(&c);
    if (r.first(if (y_bf16) 3 else 4)) try rep.control(a, "router combine", nat.got, want, tol, sh.d);
    r.note(if (y_bf16) .combine_bf16 else .combine_f32, &c);
    r.checks += 1;
}

// A fixture's words as typed values (the file's bytes are not aligned for them).
fn typed(comptime T: type, gpa: std.mem.Allocator, bytes: []const u8) ![]T {
    const out = try gpa.alloc(T, bytes.len / @sizeOf(T));
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. out.len * @sizeOf(T)]);
    return out;
}

fn file(gpu: Gpu, dir: []const u8, case: []const u8, name: []const u8) ![]u8 {
    const path = try std.fmt.allocPrint(gpu.gpa, "{s}/fixtures/{s}/{s}.bin", .{ dir, case, name });
    defer gpu.gpa.free(path);
    return std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 30));
}

fn rowsOf(case: []const u8) !usize {
    var it = std.mem.tokenizeScalar(u8, case, '-');
    while (it.next()) |t| if (std.fmt.parseInt(usize, t, 10)) |m| return m else |_| {};
    return error.BadCase;
}

// The Python engine's captured rows: real normed hidden states and the layer's real router rows, logits to combine.
fn fixtures(r: *Rig, dir: []const u8, top: std.json.ObjectMap) !void {
    const gpa = r.gpu.gpa;
    const shared: ?[]u8 = file(r.gpu, dir, "weights", "rows") catch null;
    defer if (shared) |b| gpa.free(b);
    var it = top.get("cases").?.object.iterator();
    while (it.next()) |e| {
        const case = e.key_ptr.*;
        _ = rowsOf(case) catch continue;
        if (std.mem.startsWith(u8, case, "router-")) {
            const own: ?[]u8 = file(r.gpu, dir, case, "rows") catch null;
            defer if (own) |b| gpa.free(b);
            const x = try file(r.gpu, dir, case, "x");
            defer gpa.free(x);
            const xs = try typed(u16, gpa, x);
            defer gpa.free(xs);
            const ws = try typed(u16, gpa, own orelse shared orelse return error.NoRouterRows);
            defer gpa.free(ws);
            try routerCase(r, gpa, case, xs, ws);
        } else if (std.mem.startsWith(u8, case, "topk-")) {
            const l = try file(r.gpu, dir, case, "logits");
            defer gpa.free(l);
            const ls = try typed(f32, gpa, l);
            defer gpa.free(ls);
            try selectCase(r, gpa, case, ls);
        } else if (std.mem.startsWith(u8, case, "combine-")) {
            const y = try file(r.gpu, dir, case, "y");
            defer gpa.free(y);
            const w = try file(r.gpu, dir, case, "wts");
            defer gpa.free(w);
            const ws = try typed(f32, gpa, w);
            defer gpa.free(ws);
            try combineCase(r, gpa, case, y, std.mem.startsWith(u8, case, "combine-bf16-"), ws);
        }
    }
}

const rounds = [_]usize{ 1, 2, 5, 16, 17, 40, 128, 129, 600, 1509 };

// Random rows the Python run did not draw: router rows of scale 0.03 times 4^-3..4^2, heavy-tailed inputs.
fn sweep(r: *Rig) !void {
    const gpa = r.gpu.gpa;
    const sh = r.sh;
    const ne = sh.experts + 1;
    const slots = sh.top_k + 1;
    var prng = std.Random.DefaultPrng.init(0x5eed_f640);
    const rng = prng.random();
    const w = try gpa.alloc(u16, ne * sh.d);
    defer gpa.free(w);
    for (rounds, 0..) |m, round| {
        const scale = std.math.pow(f32, 4, @floatFromInt(@as(i32, @intCast(round % 6)) - 3));
        for (w) |*v| v.* = st.bf16Of(rng.floatNorm(f32) * 0.03 * scale);
        const x = try gpa.alloc(u16, m * sh.d);
        defer gpa.free(x);
        for (x) |*v| v.* = st.bf16Of(rng.floatNorm(f32) * (if (rng.uintLessThan(u8, 10) == 0) @as(f32, 8) else 1));
        var name: [48]u8 = undefined;
        const label = try std.fmt.bufPrint(&name, "sweep {d} rows {d}", .{ round, m });
        try routerCase(r, gpa, label, x, w);
        // the same shape on a coarse grid of logits (many equal values): the pick rule
        const grid = try gpa.alloc(f32, m * ne);
        defer gpa.free(grid);
        for (grid, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(rng.intRangeAtMost(i32, -6, 6))) * 0.25;
            if (i % ne == ne - 1) v.* = @as(f32, @floatFromInt(rng.intRangeAtMost(i32, -400, 400))) * 0.25;
        }
        try selectCase(r, gpa, label, grid);
        // combine on weights the kernel produced for random logits
        const logits = try gpa.alloc(f32, m * ne);
        defer gpa.free(logits);
        for (logits) |*v| v.* = rng.floatNorm(f32) * 2;
        var lb = try r.up(std.mem.sliceAsBytes(logits));
        defer lb.free();
        var pb = try r.out(m * slots * 4);
        defer pb.free();
        var wb = try r.out(m * slots * 4);
        defer wb.free();
        try r.native.select(lb.ptr, pb.ptr, wb.ptr, m, sh);
        try r.s.synchronize();
        const wh = try check.download(r.gpu, wb);
        defer gpa.free(wh);
        const wts = try gpa.alloc(f32, m * slots);
        defer gpa.free(wts);
        for (wts, 0..) |*o, i| o.* = std.mem.bytesToValue(f32, wh[4 * i ..][0..4]);
        const y = try gpa.alloc(f32, m * slots * sh.d);
        defer gpa.free(y);
        for (y) |*v| v.* = rng.floatNorm(f32) * 3;
        const yh = try gpa.alloc(u16, y.len);
        defer gpa.free(yh);
        for (yh, y) |*o, v| o.* = st.bf16Of(v);
        try combineCase(r, gpa, label, std.mem.sliceAsBytes(y), false, wts);
        try combineCase(r, gpa, label, std.mem.sliceAsBytes(yh), true, wts);
    }
}

fn outs(a: std.mem.Allocator, sh: router.Shape) ![4]rep.Out {
    return .{
        try rep.Out.init(a, "router logits", "2^-16 sum |x w|", sh.experts + 1),
        try rep.Out.init(a, "select weights (cols 0-7 routed, 8 shared)", "2^-8 w + 2 w dl", sh.top_k + 1),
        try rep.Out.init(a, "combine fp32 y", "2^-8 |y| + 2^-20 sum |w y|", sh.d),
        try rep.Out.init(a, "combine bf16 y", "2^-8 |y| + 2^-20 sum |w y|", sh.d),
    };
}

/// `aot_dir` "native": the built-in kernels alone, no Triton comparison; `dir`: qwen35/capture_router.py's fixtures.
pub fn run(gpu: Gpu, aot_dir: []const u8, dir: ?[]const u8) !void {
    const gpa = gpu.gpa;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var set: ?cuda.aot.Set = if (std.mem.eql(u8, aot_dir, "native")) null else try cuda.aot.Set.load(gpa, gpu.io, gpu.d, gpu.ctx.device, aot_dir);
    defer if (set) |*s| s.deinit();
    const s = try cuda.Stream.init(gpu.d, true);
    var native = try router.Native.load(gpu.d, s, try gpu.ctx.capability());
    defer native.deinit();
    var r: Rig = .{ .gpu = gpu, .s = s, .native = native, .replay = if (set) |*t| .{ .set = t, .s = s } else null, .sh = .{ .d = 2048, .experts = 256, .top_k = 8 } };
    if (dir) |d| {
        const index = try std.fmt.allocPrint(gpa, "{s}/fixtures.json", .{d});
        defer gpa.free(index);
        const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, index, gpa, .limited(1 << 24));
        defer gpa.free(text);
        const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer parsed.deinit();
        const top = parsed.value.object;
        r.sh = .{ .d = @intCast(top.get("d").?.integer), .experts = @intCast(top.get("experts").?.integer), .top_k = @intCast(top.get("top_k").?.integer) };
        r.worst = try outs(arena.allocator(), r.sh);
        try fixtures(&r, d, top);
    } else r.worst = try outs(arena.allocator(), r.sh);
    try sweep(&r);
    for (&r.worst) |*w| w.table();
    check.pass("qwen-router-f64: {d} outputs within their float64 bounds{s}, {d} rows inside a near-tie", .{ r.checks, if (r.replay != null) ", native no worse than Triton" else "", r.ties });
}
