//! Qwen3.6's router: the captured Triton launches against the Python engine's bytes (qwen35/capture_router.py), the native kernels against both.

const std = @import("std");
const cuda = @import("cuda");
const router = @import("qwen35_router");
const check = @import("check.zig");

const Gpu = check.Gpu;

const Rig = struct {
    gpu: Gpu,
    s: cuda.Stream,
    replay: router.Replay,
    native: router.Native,
    sh: router.Shape,
    dir: []const u8,
    cases: usize = 0,
    bad: usize = 0,
    bytes: usize = 0,

    fn file(r: *Rig, case: []const u8, name: []const u8) ![]u8 {
        const path = try std.fmt.allocPrint(r.gpu.gpa, "{s}/fixtures/{s}/{s}.bin", .{ r.dir, case, name });
        defer r.gpu.gpa.free(path);
        return std.Io.Dir.cwd().readFileAlloc(r.gpu.io, path, r.gpu.gpa, .limited(1 << 30));
    }

    fn up(r: *Rig, bytes: []const u8) !cuda.DeviceBuffer {
        return cuda.DeviceBuffer.fromHost(r.gpu.d, bytes);
    }

    // An output the kernel must fill whole: 0xff bytes first, so an unwritten word cannot match.
    fn out(r: *Rig, n: usize) !cuda.DeviceBuffer {
        const b = try cuda.DeviceBuffer.alloc(r.gpu.d, n);
        try b.fill8(0xff, r.s.handle);
        return b;
    }

    fn same(r: *Rig, what: []const u8, case: []const u8, got: cuda.DeviceBuffer, want: []const u8) !void {
        try r.s.synchronize();
        const host = try check.download(r.gpu, got);
        defer r.gpu.gpa.free(host);
        r.cases += 1;
        r.bytes += want.len;
        check.sameBytes(what, host, want) catch {
            std.debug.print("  case {s}\n", .{case});
            var shown: usize = 0;
            var i: usize = 0;
            while (i + 4 <= host.len and shown < 4) : (i += 4) {
                if (std.mem.eql(u8, host[i..][0..4], want[i..][0..4])) continue;
                std.debug.print("    word {d}: got {x:0>8} want {x:0>8}\n", .{ i / 4, std.mem.readInt(u32, host[i..][0..4], .little), std.mem.readInt(u32, want[i..][0..4], .little) });
                shown += 1;
            }
            r.bad += 1;
        };
    }

    // The two paths on one input; `want` is the reference bytes (Python's, or the replay's).
    fn logits(r: *Rig, case: []const u8, x: []const u8, w: []const u8, want: []const u8, m: usize) !void {
        var xb = try r.up(x);
        defer xb.free();
        var wb = try r.up(w);
        defer wb.free();
        var o1 = try r.out(want.len);
        defer o1.free();
        try r.replay.logits(xb.ptr, wb.ptr, o1.ptr, m, r.sh);
        try r.same("replay logits", case, o1, want);
        var o2 = try r.out(want.len);
        defer o2.free();
        try r.native.logits(xb.ptr, wb.ptr, o2.ptr, m, r.sh);
        try r.same("native logits", case, o2, want);
        // the dispatcher: the captured launch without the native kernels, the native ones without the captured set
        inline for (.{ router.Router{ .replay = r.replay, .native = null }, router.Router{ .replay = null, .native = r.native } }) |route| {
            var o3 = try r.out(want.len);
            defer o3.free();
            try route.logits(xb.ptr, wb.ptr, o3.ptr, m, r.sh);
            try r.same("router logits", case, o3, want);
        }
        try std.testing.expectError(error.NoExactRouter, (router.Router{ .replay = null, .native = null }).logits(xb.ptr, wb.ptr, o1.ptr, m, r.sh));
    }

    fn select(r: *Rig, case: []const u8, l: []const u8, pick: []const u8, wts: []const u8, rows: usize) !void {
        var lb = try r.up(l);
        defer lb.free();
        inline for (.{ "replay", "native" }) |path| {
            var p = try r.out(pick.len);
            defer p.free();
            var w = try r.out(wts.len);
            defer w.free();
            if (comptime std.mem.eql(u8, path, "replay")) try r.replay.select(lb.ptr, p.ptr, w.ptr, rows, r.sh) else try r.native.select(lb.ptr, p.ptr, w.ptr, rows, r.sh);
            try r.same(path ++ " picks", case, p, pick);
            try r.same(path ++ " weights", case, w, wts);
        }
    }

    fn combine(r: *Rig, case: []const u8, y: []const u8, y_bf16: bool, wts: []const u8, want: []const u8, rows: usize) !void {
        var yb = try r.up(y);
        defer yb.free();
        var wb = try r.up(wts);
        defer wb.free();
        inline for (.{ "replay", "native" }) |path| {
            var o = try r.out(want.len);
            defer o.free();
            if (comptime std.mem.eql(u8, path, "replay")) try r.replay.combine(yb.ptr, y_bf16, wb.ptr, o.ptr, rows, r.sh) else try r.native.combine(yb.ptr, y_bf16, wb.ptr, o.ptr, rows, r.sh);
            try r.same(path ++ " combine", case, o, want);
        }
    }

    // The rows a case runs: its name's first number (router-16, combine-f32-16-c3).
    fn rowsOf(case: []const u8) !usize {
        var it = std.mem.tokenizeScalar(u8, case, '-');
        while (it.next()) |t| if (std.fmt.parseInt(usize, t, 10)) |m| return m else |_| {};
        return error.BadCase;
    }

    fn step(r: *Rig, comptime path: []const u8, comptime name: []const u8, x: u64, w: u64, l: u64, p: u64, wt: u64, y: u64, o: u64, m: usize) !void {
        const route = if (comptime std.mem.eql(u8, path, "replay")) router.Router{ .replay = r.replay, .native = null } else router.Router{ .replay = null, .native = r.native };
        if (comptime std.mem.eql(u8, name, "logits")) try route.logits(x, w, l, m, r.sh);
        if (comptime std.mem.eql(u8, name, "select")) try route.select(l, p, wt, m, r.sh);
        if (comptime std.mem.eql(u8, name, "combine")) try route.combine(y, false, wt, o, m, r.sh);
    }

    fn fixtureCase(r: *Rig, case: []const u8, shared_rows: ?[]const u8) !void {
        const gpa = r.gpu.gpa;
        if (std.mem.eql(u8, case, "weights")) return;
        const m = try rowsOf(case);
        if (std.mem.startsWith(u8, case, "router-")) {
            const own = r.file(case, "rows") catch null; // a layer's own router rows, else the shared ones
            defer if (own) |b| gpa.free(b);
            const x = try r.file(case, "x");
            defer gpa.free(x);
            const l = try r.file(case, "logits");
            defer gpa.free(l);
            const p = try r.file(case, "pick");
            defer gpa.free(p);
            const w = try r.file(case, "wts");
            defer gpa.free(w);
            try r.logits(case, x, own orelse shared_rows orelse return error.NoRouterRows, l, m);
            try r.select(case, l, p, w, m);
        } else if (std.mem.startsWith(u8, case, "topk-")) {
            const l = try r.file(case, "logits");
            defer gpa.free(l);
            const p = try r.file(case, "pick");
            defer gpa.free(p);
            const w = try r.file(case, "wts");
            defer gpa.free(w);
            try r.select(case, l, p, w, m);
        } else if (std.mem.startsWith(u8, case, "combine-")) {
            const y = try r.file(case, "y");
            defer gpa.free(y);
            const w = try r.file(case, "wts");
            defer gpa.free(w);
            const o = try r.file(case, "out");
            defer gpa.free(o);
            try r.combine(case, y, std.mem.startsWith(u8, case, "combine-bf16-"), w, o, m);
        }
    }
};

fn bf16(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

// Native against the replay on random inputs the Python run did not draw: scales, ties and row counts across every tile.
fn sweep(r: *Rig, rounds: usize) !void {
    const gpa = r.gpu.gpa;
    var prng = std.Random.DefaultPrng.init(0x5eed_70c4);
    const rng = prng.random();
    const ne = r.sh.experts + 1;
    const slots = r.sh.top_k + 1;
    const w = try gpa.alloc(u16, ne * r.sh.d);
    defer gpa.free(w);
    for (0..rounds) |round| {
        const m = if (round % 5 == 0) 16 * (1 + rng.uintLessThan(usize, 40)) else 1 + rng.uintLessThan(usize, 600);
        const scale = std.math.pow(f32, 4, @floatFromInt(rng.intRangeAtMost(i32, -3, 2)));
        for (w) |*v| v.* = bf16(rng.floatNorm(f32) * 0.03 * scale);
        const x = try gpa.alloc(u16, m * r.sh.d);
        defer gpa.free(x);
        for (x) |*v| v.* = bf16(rng.floatNorm(f32) * (if (rng.uintLessThan(u8, 10) == 0) @as(f32, 8) else 1));
        var xb = try r.up(std.mem.sliceAsBytes(x));
        defer xb.free();
        var wb = try r.up(std.mem.sliceAsBytes(w));
        defer wb.free();
        var lr = try r.out(m * ne * 4);
        defer lr.free();
        try r.replay.logits(xb.ptr, wb.ptr, lr.ptr, m, r.sh);
        var ln = try r.out(m * ne * 4);
        defer ln.free();
        try r.native.logits(xb.ptr, wb.ptr, ln.ptr, m, r.sh);
        try r.s.synchronize();
        const want = try check.download(r.gpu, lr);
        defer gpa.free(want);
        var case_buf: [64]u8 = undefined;
        const case = try std.fmt.bufPrint(&case_buf, "sweep {d} rows {d}", .{ round, m });
        try r.same("sweep logits", case, ln, want);
        // the same logits on a coarse grid (many equal values) for the pick rule
        const grid = try gpa.alloc(f32, m * ne);
        defer gpa.free(grid);
        for (grid, 0..) |*v, i| {
            v.* = if (round % 2 == 0) @as(f32, @floatFromInt(rng.intRangeAtMost(i32, -6, 6))) * 0.25 else std.mem.bytesAsSlice(f32, want)[i];
            if (i % ne == ne - 1) v.* = @as(f32, @floatFromInt(rng.intRangeAtMost(i32, -400, 400))) * 0.25;
        }
        var gb = try r.up(std.mem.sliceAsBytes(grid));
        defer gb.free();
        var pr = try r.out(m * slots * 4);
        defer pr.free();
        var wr = try r.out(m * slots * 4);
        defer wr.free();
        try r.replay.select(gb.ptr, pr.ptr, wr.ptr, m, r.sh);
        var pn = try r.out(m * slots * 4);
        defer pn.free();
        var wn = try r.out(m * slots * 4);
        defer wn.free();
        try r.native.select(gb.ptr, pn.ptr, wn.ptr, m, r.sh);
        try r.s.synchronize();
        const want_p = try check.download(r.gpu, pr);
        defer gpa.free(want_p);
        const want_w = try check.download(r.gpu, wr);
        defer gpa.free(want_w);
        try r.same("sweep picks", case, pn, want_p);
        try r.same("sweep weights", case, wn, want_w);
        // combine both y forms against the replay, on these weights
        const y = try gpa.alloc(f32, m * slots * r.sh.d);
        defer gpa.free(y);
        for (y) |*v| v.* = rng.floatNorm(f32) * 3;
        const yh = try gpa.alloc(u16, y.len);
        defer gpa.free(yh);
        for (yh, y) |*d, v| d.* = bf16(v);
        for ([_]bool{ false, true }) |as_bf16| {
            var yb = try r.up(if (as_bf16) std.mem.sliceAsBytes(yh) else std.mem.sliceAsBytes(y));
            defer yb.free();
            var o1 = try r.out(m * r.sh.d * 2);
            defer o1.free();
            try r.replay.combine(yb.ptr, as_bf16, wr.ptr, o1.ptr, m, r.sh);
            var o2 = try r.out(m * r.sh.d * 2);
            defer o2.free();
            try r.native.combine(yb.ptr, as_bf16, wr.ptr, o2.ptr, m, r.sh);
            try r.s.synchronize();
            const wc = try check.download(r.gpu, o1);
            defer gpa.free(wc);
            try r.same("sweep combine", case, o2, wc);
        }
    }
}

// Microseconds a router step takes back to back at `m` rows, replay against native (information, not a gate).
fn timing(r: *Rig, m: usize) !void {
    const ne = r.sh.experts + 1;
    const slots = r.sh.top_k + 1;
    var xb = try r.out(m * r.sh.d * 2);
    defer xb.free();
    try r.s.synchronize();
    var wb = try r.out(ne * r.sh.d * 2);
    defer wb.free();
    var lb = try r.out(m * ne * 4);
    defer lb.free();
    var pb = try r.out(m * slots * 4);
    defer pb.free();
    var wt = try r.out(m * slots * 4);
    defer wt.free();
    var yb = try r.out(m * slots * r.sh.d * 4);
    defer yb.free();
    var ob = try r.out(m * r.sh.d * 2);
    defer ob.free();
    try r.s.synchronize();
    // finite inputs: a router over zeros, ties everywhere
    try xb.fill8(0, r.s.handle);
    try wb.fill8(0, r.s.handle);
    try lb.fill8(0, r.s.handle);
    try yb.fill8(0, r.s.handle);
    try wt.fill8(0, r.s.handle);
    var start = try cuda.Event.init(r.gpu.d, true);
    defer start.deinit();
    var end = try cuda.Event.init(r.gpu.d, true);
    defer end.deinit();
    const reps = 200;
    inline for (.{ "replay", "native" }) |path| {
        inline for (.{ "logits", "select", "combine" }) |step| {
            for (0..10) |_| try r.step(path, step, xb.ptr, wb.ptr, lb.ptr, pb.ptr, wt.ptr, yb.ptr, ob.ptr, m);
            try start.record(r.s);
            for (0..reps) |_| try r.step(path, step, xb.ptr, wb.ptr, lb.ptr, pb.ptr, wt.ptr, yb.ptr, ob.ptr, m);
            try end.record(r.s);
            try end.synchronize();
            std.debug.print("  {d:>5} rows {s:<6} {s:<7} {d:>7.2} us\n", .{ m, path, step, @as(f64, try cuda.Event.elapsedMs(start, end)) * 1000 / reps });
        }
    }
}

pub fn run(gpu: Gpu, aot_dir: []const u8, dir: []const u8, rounds: usize) !void {
    const gpa = gpu.gpa;
    var set = try cuda.aot.Set.load(gpa, gpu.io, gpu.d, gpu.ctx.device, aot_dir);
    defer set.deinit();
    const s = try cuda.Stream.init(gpu.d, true);
    var native = try router.Native.load(gpu.d, s, try gpu.ctx.capability());
    defer native.deinit();
    const index_path = try std.fmt.allocPrint(gpa, "{s}/fixtures.json", .{dir});
    defer gpa.free(index_path);
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, index_path, gpa, .limited(1 << 24));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const top = parsed.value.object;
    var r: Rig = .{
        .gpu = gpu,
        .s = s,
        .replay = .{ .set = &set, .s = s },
        .native = native,
        .sh = .{ .d = @intCast(top.get("d").?.integer), .experts = @intCast(top.get("experts").?.integer), .top_k = @intCast(top.get("top_k").?.integer) },
        .dir = dir,
    };
    const rows_w: ?[]u8 = r.file("weights", "rows") catch null;
    defer if (rows_w) |b| gpa.free(b);
    var it = top.get("cases").?.object.iterator();
    while (it.next()) |e| try r.fixtureCase(e.key_ptr.*, rows_w);
    std.debug.print("fixtures: {d} comparisons, {d} bytes, {d} differ\n", .{ r.cases, r.bytes, r.bad });
    const fixture_bad = r.bad;
    const before = r.cases;
    try sweep(&r, rounds);
    std.debug.print("sweep: {d} comparisons, {d} differ (fixtures {d})\n", .{ r.cases - before, r.bad - fixture_bad, fixture_bad });
    for ([_]usize{ 1, 16, 256 }) |m| try timing(&r, m);
    try check.expect(r.bad == 0, "{d} of {d} comparisons differ", .{ r.bad, r.cases });
    check.pass("qwen35-router: replay and native byte-equal to the Python engine over {d} comparisons", .{r.cases});
}
