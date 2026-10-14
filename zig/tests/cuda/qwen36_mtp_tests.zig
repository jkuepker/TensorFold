//! The Qwen3.6 MTP head against the Python engine's dumps (qwen36/capture.py --mtp), byte for byte: every call's norms,
//! normed rows and draft logits, replayed in the order the Python engine made them over the head's own cache.

const std = @import("std");
const cuda = @import("cuda");
const qwen27 = @import("qwen27");
const qwen35 = @import("qwen35_cuda");
const check = @import("check.zig");

const bf = 2;
const native = qwen35.native;
const mtp = qwen35.mtp;

const Run = struct {
    gpu: check.Gpu,
    head: *mtp.Head,
    dir: []const u8 = "", // the call's dump folder
    bad: usize = 0,
    seen: usize = 0,
    first: ?[]const u8 = null,
    arena: std.heap.ArenaAllocator,

    fn ops(r: *Run) qwen27.kernels.Ops {
        return r.head.e.forward().ops;
    }

    fn file(r: *Run, comptime fmt: []const u8, args: anytype) !?[]u8 {
        const path = try std.fmt.allocPrint(r.gpu.gpa, fmt, args);
        defer r.gpu.gpa.free(path);
        return std.Io.Dir.cwd().readFileAlloc(r.gpu.io, path, r.gpu.gpa, .limited(1 << 31)) catch null;
    }

    /// Device bytes `len` at `ptr` against the dump `name` of the current call; missing dumps are skipped.
    fn compare(r: *Run, label: []const u8, ptr: u64, len: usize, name: []const u8) !void {
        const want = (try r.file("{s}/{s}.bin", .{ r.dir, name })) orelse return;
        defer r.gpu.gpa.free(want);
        const got = try r.gpu.gpa.alloc(u8, len);
        defer r.gpu.gpa.free(got);
        const o = r.ops();
        try o.s.synchronize();
        try o.download(got, ptr);
        try o.s.synchronize();
        r.seen += 1;
        if (want.len == len and std.mem.eql(u8, got, want)) return;
        r.bad += 1;
        if (r.first == null) {
            var at: usize = 0;
            while (at < @min(want.len, len) and got[at] == want[at]) at += 1;
            r.first = try std.fmt.allocPrint(r.arena.allocator(), "{s} {s}: {d} bytes (dump {d}), first difference at byte {d}", .{ r.dir, label, len, want.len, at });
        }
    }

    fn norm(ctx: *anyopaque, index: usize, h: u64, y: u64, xs: u64, rows: usize) anyerror!void {
        const r: *Run = @ptrCast(@alignCast(ctx));
        const hid = r.head.e.w.g.hidden;
        var buf: [32]u8 = undefined;
        inline for (.{ .{ "h", h, hid * bf }, .{ "y", y, hid * bf }, .{ "xs", xs, hid / 64 * 4 } }) |part| {
            const name = try std.fmt.bufPrint(&buf, "{d:0>3}_{s}", .{ index, part[0] });
            try r.compare(name, part[1], rows * part[2], name);
        }
    }
};

fn ids(gpa: std.mem.Allocator, v: std.json.Value) ![]u32 {
    const out = try gpa.alloc(u32, v.array.items.len);
    for (out, v.array.items) |*o, x| o.* = @intCast(x.integer);
    return out;
}

pub fn run(gpu: check.Gpu, model: []const u8, kernels: []const u8, oracle: []const u8, only: []const u8) !void {
    const gpa = gpu.gpa;
    const e = try native.engine(gpa, gpu.io, gpu.ctx, model, if (std.mem.eql(u8, kernels, "native")) null else kernels, 8192);
    defer e.deinit();
    const moe: *native.Moe = @ptrCast(@alignCast(e.hooks.?.ctx));
    var pk = try qwen35.pack.Kernels.load(gpu.d, e.s, (try gpu.ctx.attribute(.integrated)) == 0);
    const w = try qwen35.mtp_weights.load(gpa, gpu.io, &pk, model, e.w.g);
    pk.deinit();
    var head = try mtp.Head.init(gpa, e, w, &moe.block);
    defer head.deinit();
    var r: Run = .{ .gpu = gpu, .head = &head, .arena = .init(gpa) };
    defer r.arena.deinit();
    head.trace = .{ .ctx = &r, .norm = Run.norm };
    const ops = e.forward().ops;
    const hid = e.w.g.hidden;
    var prompts = std.mem.tokenizeScalar(u8, only, ',');
    var failed: usize = 0;
    while (prompts.next()) |prompt| {
        for ([_][]const u8{"greedy"}) |kind| {
            const root = try std.fmt.allocPrint(r.arena.allocator(), "{s}/{s}/{s}", .{ oracle, prompt, kind });
            const text = (try r.file("{s}/calls.json", .{root})) orelse return error.NoCalls;
            defer gpa.free(text);
            var calls = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
            defer calls.deinit();
            var cache = try mtp.Cache.init(gpu.d, e.w.g, 8192);
            defer cache.deinit();
            r.bad = 0;
            r.seen = 0;
            r.first = null;
            for (calls.value.array.items) |call| {
                const o = call.object;
                const dir = o.get("dir").?.string;
                r.dir = try std.fmt.allocPrint(r.arena.allocator(), "{s}/{s}", .{ root, dir });
                const tokens = try ids(gpa, o.get("tokens").?);
                defer gpa.free(tokens);
                const p0: usize = @intCast(o.get("p0").?.integer);
                const states = (try r.file("{s}/states.bin", .{r.dir})) orelse return error.NoStates;
                defer gpa.free(states);
                if (states.len != tokens.len * hid * bf) return error.BadStates;
                var dev = try cuda.DeviceBuffer.alloc(gpu.d, states.len);
                defer dev.free();
                try ops.upload(dev.ptr, states);
                try ops.s.synchronize();
                try head.forward(&cache, dev.ptr, tokens, p0);
                try r.compare("normed", head.normed, tokens.len * hid * bf, "out");
                if (o.get("logits").?.bool) {
                    try head.scores(tokens.len - 1);
                    try r.compare("logits", head.logits, head.w.ids.len * bf, "logits");
                }
            }
            if (r.bad == 0) {
                check.pass("{s}/{s}: {d} head calls, {d} dumps byte-equal", .{ prompt, kind, calls.value.array.items.len, r.seen });
            } else {
                failed += 1;
                std.debug.print("FAIL {s}/{s}: {d} of {d} dumps differ; first: {s}\n", .{ prompt, kind, r.bad, r.seen, r.first.? });
            }
        }
    }
    try check.expect(failed == 0, "{d} prompts differ from the oracle", .{failed});
}
