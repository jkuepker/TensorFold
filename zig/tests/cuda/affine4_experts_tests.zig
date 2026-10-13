//! Grouped MLX affine-4 experts against recorded fixture bytes (the Python engine's): packing, plan, SwiGLU and relu^2 up, down.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Fixture = @import("fixture.zig").Fixture;
const Gpu = check.Gpu;

const grouped = cuda.grouped;
const ex_ = cuda.experts;
const A4 = ex_.Affine4Experts;

fn param(fx: Fixture, name: []const u8) !usize {
    return @intCast(try fx.int(name));
}

fn arr(fx: Fixture, comptime fmt: []const u8, args: anytype) ![]u8 {
    var name: [48]u8 = undefined;
    return fx.bytes(try std.fmt.bufPrint(&name, fmt, args));
}

/// A fixture projection's words, scales and biases on the device.
fn mats(gpu: Gpu, fx: Fixture, comptime name: []const u8, bufs: *std.ArrayList(cuda.DeviceBuffer)) !A4.Mats {
    var ptrs: [3]u64 = undefined;
    inline for (.{ "words", "scales", "biases" }, 0..) |part, i| {
        const bytes = try fx.bytes(name ++ "_" ++ part);
        defer gpu.gpa.free(bytes);
        const b = try cuda.DeviceBuffer.fromHost(gpu.d, bytes);
        try bufs.append(gpu.gpa, b);
        ptrs[i] = b.ptr;
    }
    return .{ .words = ptrs[0], .scales = ptrs[1], .biases = ptrs[2] };
}

pub fn experts(gpu: Gpu, dir: []const u8) !void {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    const a = gpu.gpa;
    const e = try param(fx, "experts");
    const d = try param(fx, "dims");
    const ni = try param(fx, "width");
    const slots = try param(fx, "slots");
    const limit: f32 = @floatCast(try fx.float("limit"));

    var bufs: std.ArrayList(cuda.DeviceBuffer) = .empty;
    defer {
        for (bufs.items) |*b| b.free();
        bufs.deinit(a);
    }
    var stream = try cuda.Stream.init(gpu.d, false); // blocking: launches wait for fromHost's legacy-stream copies
    defer stream.deinit();
    var mods: [3]cuda.Module = undefined;
    for ([_][]const u8{ cuda.kernels.experts, cuda.kernels.experts_prefill, cuda.kernels.experts_pack }, &mods) |blob, *m| m.* = try cuda.Module.load(gpu.d, blob);
    defer for (&mods) |*m| m.unload();
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    const ex = try A4.resolve(mods[0], mods[1], mods[2], sms);
    const router = try grouped.Router.resolve(mods[0]);

    // packing: gate then up stacked per block, the down and the lone relu^2 up as make() lays them out
    const gate = try mats(gpu, fx, "gate", &bufs);
    const up = try mats(gpu, fx, "up", &bufs);
    const down_m = try mats(gpu, fx, "down", &bufs);
    const one = A4.bytes(e, ni, d);
    const packed_up = try cuda.DeviceBuffer.alloc(gpu.d, 2 * one);
    try bufs.append(a, packed_up);
    const packed_relu = try cuda.DeviceBuffer.alloc(gpu.d, one);
    try bufs.append(a, packed_relu);
    const packed_down = try cuda.DeviceBuffer.alloc(gpu.d, A4.bytes(e, d, ni));
    try bufs.append(a, packed_down);
    const scratch = try cuda.DeviceBuffer.alloc(gpu.d, one);
    try bufs.append(a, scratch);
    try ex.packGateUp(gpu.d, stream, gate, up, scratch.ptr, packed_up.ptr, e, ni, d);
    try ex.pack(stream, gate.words, gate.scales, gate.biases, packed_relu.ptr, e, ni, d);
    try ex.pack(stream, down_m.words, down_m.scales, down_m.biases, packed_down.ptr, e, d, ni);
    try stream.synchronize();
    inline for (.{ .{ "packed_up", &packed_up }, .{ "packed_relu2_up", &packed_relu }, .{ "packed_down", &packed_down } }) |c| {
        const got = try check.download(gpu, c[1].*);
        defer a.free(got);
        const want = try fx.bytes(c[0]);
        defer a.free(want);
        try check.sameBytes(c[0], got, want);
    }
    check.pass("affine4 experts: {d} x [{d}, {d}] gate-up (stacked), relu^2 up and down packed as make() packs them", .{ e, ni, d });

    const swiglu: ex_.Layer = .{ .format = .affine4, .swiglu = true, .up = packed_up.ptr, .down = packed_down.ptr, .width = ni, .dims = d, .experts = e };
    var limited = swiglu;
    limited.limit = limit;
    var relu2 = swiglu;
    relu2.swiglu = false;
    relu2.up = packed_relu.ptr;
    const layers = [_]struct { name: []const u8, l: ex_.Layer }{ .{ .name = "swiglu", .l = swiglu }, .{ .name = "limited", .l = limited }, .{ .name = "relu2", .l = relu2 } };
    const g: ex_.Grouped = .{ .affine4 = &ex };
    inline for (.{ .{ "decode", false }, .{ "prompt", true } }) |set| {
        var it = std.mem.tokenizeScalar(u8, try fx.string(set[0]), ',');
        while (it.next()) |tok| {
            const rows = try std.fmt.parseInt(usize, tok, 10);
            try oneCase(gpu, fx, router, g, stream, &layers, rows, slots, set[1]);
        }
    }
}

fn oneCase(gpu: Gpu, fx: Fixture, router: grouped.Router, g: ex_.Grouped, s: cuda.Stream, layers: anytype, rows: usize, slots: usize, prompt: bool) !void {
    const a = gpu.gpa;
    const tag: u8 = if (prompt) 'p' else 'd';
    const experts_n = layers[0].l.experts;
    const pairs = rows * slots;
    const picks = try arr(fx, "{c}{d}_picks", .{ tag, rows });
    defer a.free(picks);
    var dpicks = try cuda.DeviceBuffer.fromHost(gpu.d, picks);
    defer dpicks.free();
    const t: usize = if (prompt) ex_.promptTile(.affine4) else 16;
    const wide = pairs > grouped.small;
    const n_max = grouped.maxItems(pairs, experts_n, 16);
    var bufs: [5]cuda.DeviceBuffer = undefined;
    const sizes = [5]usize{ pairs * 4, n_max * 12, 8, (if (wide) pairs else 1) * 4, (if (wide) (pairs + 1023) / 1024 * experts_n else 1) * 4 };
    var made: usize = 0;
    defer for (bufs[0..made]) |*b| b.free();
    for (sizes, &bufs) |n, *b| {
        b.* = try cuda.DeviceBuffer.alloc(gpu.d, n);
        made += 1;
    }
    const plan: grouped.Plan = .{ .members = bufs[0].ptr, .items = bufs[1].ptr, .counts = bufs[2].ptr, .rank = bufs[3].ptr, .hist = bufs[4].ptr };
    try router.route(s, dpicks.ptr, pairs, experts_n, t, plan);
    try s.synchronize();

    const counts = try check.download(gpu, bufs[2]);
    defer a.free(counts);
    const want_counts = try arr(fx, "{c}{d}_counts", .{ tag, rows });
    defer a.free(want_counts);
    try check.sameBytes("plan counts", counts, want_counts);
    const n_items: usize = @intCast(std.mem.bytesToValue(i32, counts[0..4]));
    const items = try check.download(gpu, bufs[1]);
    defer a.free(items);
    const want_items = try arr(fx, "{c}{d}_items", .{ tag, rows });
    defer a.free(want_items);
    try check.sameBytes("plan items", items[0 .. n_items * 12], want_items[0 .. n_items * 12]);
    const members = try check.download(gpu, bufs[0]);
    defer a.free(members);
    const want_members = try arr(fx, "{c}{d}_members", .{ tag, rows });
    defer a.free(want_members);
    try check.sameBytes("plan members", members, want_members);

    const x = try arr(fx, "{c}{d}_x", .{ tag, rows });
    defer a.free(x);
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, x);
    defer dx.free();
    const bound = grouped.maxItems(pairs, experts_n, t);
    for (layers) |lay| {
        const l = lay.l;
        var what: [48]u8 = undefined;
        var act = try cuda.DeviceBuffer.alloc(gpu.d, pairs * l.width * 2);
        defer act.free();
        try act.fill8(0xff, s.handle); // a call that writes nothing fails rather than pass on old bytes
        const up_in: ex_.Rows = .{ .x = dx.ptr, .stride = l.dims, .slots = slots };
        if (prompt) try g.prompt(s, .up, up_in, l, plan, act.ptr, bound, -1) else try g.decode(s, .up, up_in, l, plan, act.ptr, bound, -1);
        try s.synchronize();
        const got_act = try check.download(gpu, act);
        defer a.free(got_act);
        const want_act = try arr(fx, "{c}{d}_{s}_act", .{ tag, rows, lay.name });
        defer a.free(want_act);
        try check.sameBytes(try std.fmt.bufPrint(&what, "{s} gate-up", .{lay.name}), got_act, want_act);

        var y = try cuda.DeviceBuffer.alloc(gpu.d, pairs * l.dims * @as(usize, if (prompt) 2 else 4));
        defer y.free();
        try y.fill8(0xff, s.handle);
        const down_in: ex_.Rows = .{ .x = act.ptr, .stride = l.width };
        if (prompt) try g.prompt(s, .down_bf16, down_in, l, plan, y.ptr, bound, -1) else try g.decode(s, .down_f32, down_in, l, plan, y.ptr, bound, -1);
        try s.synchronize();
        const got_y = try check.download(gpu, y);
        defer a.free(got_y);
        const want_y = try arr(fx, "{c}{d}_{s}_y", .{ tag, rows, lay.name });
        defer a.free(want_y);
        try check.sameBytes(try std.fmt.bufPrint(&what, "{s} down", .{lay.name}), got_y, want_y);
    }
    check.pass("affine4 experts: {d} rows x {d} slots ({s}, {d} items) plan, swiglu, clamped swiglu and relu^2 up, down equal Python's bytes", .{ rows, slots, if (prompt) "prompt" else "decode", n_items });
}
