//! The Qwen3.6 engine against the Python engine's dumps (qwen36/capture.py), byte for byte: norms, MoE layers, logits, state.

const std = @import("std");
const cuda = @import("cuda");
const qwen27 = @import("qwen27");
const qwen35 = @import("qwen35_cuda");
const check = @import("check.zig");

const Forward = qwen27.Forward;
const st = qwen27.state;
const bf = 2;
const Sha256 = std.crypto.hash.sha2.Sha256;

/// What a dump holds, and how its elements read when two differ.
const Kind = enum { h, y, xs, router, pick, wts, moe_out, logits, rec, conv, keys, values };
const kinds = std.enums.values(Kind);
const n_kinds = kinds.len;
const Dtype = enum { bf16, f32, i32 };

fn dtypeOf(k: Kind) Dtype {
    return switch (k) {
        .h, .y, .moe_out, .logits, .conv, .keys, .values => .bf16,
        .xs, .router, .wts, .rec => .f32,
        .pick => .i32,
    };
}

/// Tensors compared, equal, and the first difference of each kind and layer.
const Tally = struct {
    total: [n_kinds][41]usize = @splat(@splat(0)),
    equal: [n_kinds][41]usize = @splat(@splat(0)),
    first: [n_kinds]?[]const u8 = @splat(null),
    worst: [n_kinds]f64 = @splat(0),
};

const Run = struct {
    gpu: check.Gpu,
    f: *Forward,
    block: *qwen35.moe.Block,
    w: *const qwen35.weights.Weights,
    teacher: bool,
    oracle: []const u8, // the dump tree's root
    dir: []const u8 = "", // the pass's dump folder
    label: []const u8 = "", // what the pass is, for messages
    moes: usize = 0,
    digests: ?std.json.Parsed(std.json.Value) = null,
    tally: Tally = .{},
    arena: std.heap.ArenaAllocator,

    fn ops(r: *Run) qwen27.kernels.Ops {
        return r.f.ops;
    }

    /// Device bytes, downloaded.
    fn fetch(r: *Run, dev: u64, len: usize) ![]u8 {
        const out = try r.gpu.gpa.alloc(u8, len);
        errdefer r.gpu.gpa.free(out);
        try r.ops().s.synchronize();
        try r.ops().download(out, dev);
        try r.ops().s.synchronize();
        return out;
    }

    fn element(k: Dtype, b: []const u8, i: usize) f64 {
        return switch (k) {
            .bf16 => @as(f32, @bitCast(@as(u32, std.mem.readInt(u16, b[i * 2 ..][0..2], .little)) << 16)),
            .f32 => @as(f32, @bitCast(std.mem.readInt(u32, b[i * 4 ..][0..4], .little))),
            .i32 => @floatFromInt(std.mem.readInt(i32, b[i * 4 ..][0..4], .little)),
        };
    }

    /// The expectation: the dump file, or the digest the long prompt keeps instead.
    fn expected(r: *Run, name: []const u8) !union(enum) { bytes: []u8, digest: [32]u8, none } {
        const gpa = r.gpu.gpa;
        const path = try std.fmt.allocPrint(gpa, "{s}/{s}.bin", .{ r.dir, name });
        defer gpa.free(path);
        if (std.Io.Dir.cwd().readFileAlloc(r.gpu.io, path, gpa, .limited(1 << 31))) |bytes| return .{ .bytes = bytes } else |_| {}
        if (r.digests) |d| if (d.value.object.get(name)) |e| {
            var out: [32]u8 = undefined;
            _ = try std.fmt.hexToBytes(&out, e.array.items[2].string);
            return .{ .digest = out };
        };
        return .none;
    }

    /// One tensor of the pass against its dump; `layer` files the result, `teach` puts the dump's bytes in its place.
    fn compare(r: *Run, kind: Kind, layer: usize, name: []const u8, dev: u64, len: usize, teach: bool) !void {
        const gpa = r.gpu.gpa;
        const got = try r.fetch(dev, len);
        defer gpa.free(got);
        const want = try r.expected(name);
        defer switch (want) {
            .bytes => |b| gpa.free(b),
            else => {},
        };
        const k = @intFromEnum(kind);
        const l = @min(layer, 40);
        r.tally.total[k][l] += 1;
        var same = false;
        var why: []const u8 = "";
        switch (want) {
            .none => why = "no dump",
            .digest => |d| {
                var h: [32]u8 = undefined;
                Sha256.hash(got, &h, .{});
                same = std.mem.eql(u8, &h, &d);
                why = "digest differs";
            },
            .bytes => |b| {
                if (b.len != got.len) {
                    why = "size differs";
                } else {
                    same = std.mem.eql(u8, got, b);
                    if (!same) {
                        const dt = dtypeOf(kind);
                        const n = got.len / @as(usize, if (dt == .bf16) 2 else 4);
                        var worst: f64 = 0;
                        var count: usize = 0;
                        var at: usize = n;
                        for (0..n) |i| {
                            const a = element(dt, got, i);
                            const e = element(dt, b, i);
                            if (a != e and !(std.math.isNan(a) and std.math.isNan(e))) {
                                count += 1;
                                if (at == n) at = i;
                            }
                            const diff = @abs(a - e);
                            if (diff > worst) worst = diff;
                        }
                        r.tally.worst[k] = @max(r.tally.worst[k], worst);
                        why = try std.fmt.allocPrint(r.arena.allocator(), "{d} of {d} elements differ, first {d}, max abs {e}", .{ count, n, at, worst });
                    }
                }
            },
        }
        if (same) {
            r.tally.equal[k][l] += 1;
            return;
        }
        if (r.tally.first[k] == null) {
            const msg = try std.fmt.allocPrint(r.arena.allocator(), "{s} {s} (layer {d}): {s}", .{ r.label, name, layer, why });
            r.tally.first[k] = msg;
            std.debug.print("DIFF {s}\n", .{msg});
        }
        if (teach and r.teacher) switch (want) {
            .bytes => |b| if (b.len == len) {
                try r.ops().upload(dev, b);
                try r.ops().s.synchronize();
            },
            else => {},
        };
    }

    fn tag(r: *Run, comptime fmt: []const u8, args: anytype) ![]u8 {
        return std.fmt.allocPrint(r.arena.allocator(), fmt, args);
    }

    /// add_rmsnorm call `n` of the pass: two a layer, then the final norm.
    fn norm(ctx: *anyopaque, n: usize, h: u64, y: u64, xs: u64, rows: usize) anyerror!void {
        const r: *Run = @ptrCast(@alignCast(ctx));
        const g = r.w.base.g;
        const layer = @min(n % 81 / 2, g.layers);
        const hid = g.hidden;
        try r.compare(.h, layer, try r.tag("{d:0>3}_h", .{n}), h, rows * hid * bf, true);
        try r.compare(.y, layer, try r.tag("{d:0>3}_y", .{n}), y, rows * hid * bf, true);
        try r.compare(.xs, layer, try r.tag("{d:0>3}_xs", .{n}), xs, rows * hid / 64 * 4, true);
    }

    /// The MoE layer in place of the dense MLP, then its dumps: router logits, picks, weights and output.
    fn mlp(ctx: *anyopaque, layer: usize, y: u64, xs: u64, rows: usize, prompt: bool, out: u64) anyerror!void {
        _ = xs;
        const r: *Run = @ptrCast(@alignCast(ctx));
        const g = r.w.base.g;
        const slots = g.moe.?.top_k + 1;
        try r.block.run(r.w.layers[layer].moe, y, rows, prompt, out);
        const m = r.moes;
        r.moes += 1;
        try r.compare(.router, layer, try r.tag("m{d:0>2}_logits", .{m}), r.block.logits(), rows * r.block.count * 4, false);
        try r.compare(.pick, layer, try r.tag("m{d:0>2}_pick", .{m}), r.block.picks(), rows * slots * 4, false);
        try r.compare(.wts, layer, try r.tag("m{d:0>2}_wts", .{m}), r.block.weights(), rows * slots * 4, false);
        try r.compare(.moe_out, layer, try r.tag("m{d:0>2}_out", .{m}), out, rows * g.hidden * bf, true);
    }

    fn begin(r: *Run, comptime fmt: []const u8, args: anytype, digests: bool) !void {
        const a = r.arena.allocator();
        r.dir = try std.fmt.allocPrint(a, fmt, args);
        r.label = r.dir[@min(r.oracle.len + 1, r.dir.len)..];
        r.f.norms = 0;
        r.moes = 0;
        if (r.digests) |*d| d.deinit();
        r.digests = null;
        if (digests) {
            const path = try std.fmt.allocPrint(a, "{s}/digests.json", .{r.dir});
            const text = try std.Io.Dir.cwd().readFileAlloc(r.gpu.io, path, a, .limited(1 << 26));
            r.digests = try std.json.parseFromSlice(std.json.Value, r.gpu.gpa, text, .{});
        }
    }
};

fn readJson(gpu: check.Gpu, path: []const u8) !std.json.Parsed(std.json.Value) {
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 28));
    defer gpu.gpa.free(text);
    return std.json.parseFromSlice(std.json.Value, gpu.gpa, text, .{});
}

fn ids(gpa: std.mem.Allocator, v: std.json.Value) ![]u32 {
    const out = try gpa.alloc(u32, v.array.items.len);
    for (out, v.array.items) |*o, x| o.* = @intCast(x.integer);
    return out;
}

/// Free-running (the engine's own values flow on) or `teacher` (each compared value is replaced by the dump's).
pub fn run(gpu: check.Gpu, model: []const u8, kernels: []const u8, oracle: []const u8, only: []const u8, teacher: bool, steps: usize) !void {
    const gpa = gpu.gpa;
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const discrete = try gpu.ctx.attribute(.integrated) == 0;
    var packk = try qwen35.pack.Kernels.load(gpu.d, stream, discrete);
    defer packk.deinit();
    var w = try qwen35.weights.load(gpa, gpu.io, &packk, model);
    defer w.deinit();
    try stream.synchronize();
    const g = w.base.g;
    var k = try qwen27.kernels.Kernels.load(gpa, gpu.io, gpu.ctx, kernels, g.linear_v_heads);
    defer k.deinit();
    const prompt_rows = 4096;
    const capacity = 8192;
    var scratch = try st.Scratch.init(gpa, gpu.d, g, prompt_rows, capacity);
    defer scratch.deinit();
    var block = try qwen35.moe.Block.load(gpu.ctx, stream, g, prompt_rows, &k.triton);
    defer block.deinit();
    var f: Forward = .{ .gpa = gpa, .w = &w.base, .ops = .{ .k = &k, .s = stream }, .t = k.tri(stream), .s = &scratch };
    var run_: Run = .{ .gpu = gpu, .f = &f, .block = &block, .w = &w, .teacher = teacher, .oracle = oracle, .arena = .init(gpa) };
    defer run_.arena.deinit();
    f.hooks = .{ .ctx = &run_, .mlp = Run.mlp, .norm = Run.norm };

    var names = std.mem.tokenizeScalar(u8, only, ',');
    var failed = false;
    while (names.next()) |prompt| {
        runPrompt(&run_, oracle, prompt, steps) catch |e| {
            std.debug.print("FAIL {s}: {s}\n", .{ prompt, @errorName(e) });
            failed = true;
        };
    }
    report(&run_.tally);
    for (run_.tally.first) |x| {
        if (x != null) failed = true;
    }
    if (failed) return error.TestFailed;
    check.pass("qwen36 engine byte-equal to the Python oracle on {s}", .{only});
}

fn runPrompt(r: *Run, oracle: []const u8, prompt: []const u8, steps: usize) !void {
    const gpu = r.gpu;
    const gpa = gpu.gpa;
    const f = r.f;
    const a = r.arena.allocator();
    const root = try std.fmt.allocPrint(a, "{s}/{s}", .{ oracle, prompt });
    var pj = try readJson(gpu, try std.fmt.allocPrint(a, "{s}/prompts.json", .{oracle}));
    defer pj.deinit();
    var tj = try readJson(gpu, try std.fmt.allocPrint(a, "{s}/tokens.json", .{root}));
    defer tj.deinit();
    const tokens = try ids(gpa, pj.value.object.get(prompt) orelse return error.UnknownPrompt);
    defer gpa.free(tokens);
    const g = r.w.base.g;
    const digests = std.mem.eql(u8, prompt, "long"); // its prefill keeps digests, not tensors

    // one-row steps from an empty state, forcing the prompt's tokens
    {
        const forced = try ids(gpa, tj.value.object.get("cold_forced").?);
        defer gpa.free(forced);
        const want = try ids(gpa, tj.value.object.get("cold_sampled").?);
        defer gpa.free(want);
        var seq = try st.Seq.init(gpa, f.ops, g, 8192);
        defer seq.deinit();
        try step(r, &seq, root, "cold", forced[0..@min(forced.len, steps)], want);
    }

    // the prompt in the chunks prefill_state cuts, then its committed state
    var seq = try st.Seq.init(gpa, f.ops, g, 8192);
    defer seq.deinit();
    try r.begin("{s}/prefill", .{root}, digests);
    const n_chunks = (tokens.len + 4095) / 4096;
    for (0..n_chunks) |j| {
        const lo = tokens.len * j / n_chunks;
        const hi = tokens.len * (j + 1) / n_chunks;
        try f.chunk(tokens[lo..hi], &seq, j + 1 == n_chunks);
    }
    try f.ops.s.synchronize();
    try state(r, &seq, try std.fmt.allocPrint(a, "{s}/state", .{root}));
    try f.head(&seq);
    var pick: [1]u32 = undefined;
    try f.picks(1, &pick);
    const first: u32 = @intCast(tj.value.object.get("first").?.integer);
    try check.expect(pick[0] == first, "{s}: prefill's first token {d}, oracle {d}", .{ prompt, pick[0], first });

    // teacher-forced steps after the prefill
    {
        const forced = try ids(gpa, tj.value.object.get("warm_forced").?);
        defer gpa.free(forced);
        const want = try ids(gpa, tj.value.object.get("warm_sampled").?);
        defer gpa.free(want);
        try step(r, &seq, root, "warm", forced[0..@min(forced.len, steps)], want);
    }
}

/// One-row steps of `seq`: each forced token's window, its logits against logits.bin, the greedy pick, the commit.
fn step(r: *Run, seq: *st.Seq, root: []const u8, phase: []const u8, forced: []const u32, sampled: []const u32) !void {
    const f = r.f;
    const g = r.w.base.g;
    for (forced, 0..) |tok, i| {
        try r.begin("{s}/{s}/step{d:0>3}", .{ root, phase, i }, false);
        const parts = [_]qwen27.Part{.{ .seq = seq, .tokens = &.{tok} }};
        try f.round(&parts);
        try r.compare(.logits, g.layers, "logits", f.s.logits, qwen27.config.vocab * bf, false);
        var pick: [1]u32 = undefined;
        try f.picks(1, &pick);
        try check.expect(pick[0] == sampled[i], "{s} step {d}: greedy token {d}, oracle {d}", .{ phase, i, pick[0], sampled[i] });
        try f.commit(&parts, &.{&.{0}});
    }
}

/// The committed state after a prefill: each DeltaNet layer's recurrence and conv window, each attention layer's cache rows.
fn state(r: *Run, seq: *const st.Seq, dir: []const u8) !void {
    const g = r.w.base.g;
    try r.begin("{s}", .{dir}, false);
    for (0..g.layers) |i| {
        if (qwen27.config.linear(i)) {
            try r.compare(.rec, i, try r.tag("L{d:0>2}_rec", .{i}), seq.rec[i], g.deltaBytes(), false);
            try r.compare(.conv, i, try r.tag("L{d:0>2}_conv", .{i}), seq.conv[i], g.convBytes(), false);
        } else {
            try r.compare(.keys, i, try r.tag("L{d:0>2}_k", .{i}), seq.keys[i], seq.pos * g.kvInner() * bf, false);
            try r.compare(.values, i, try r.tag("L{d:0>2}_v", .{i}), seq.values[i], seq.pos * g.kvInner() * bf, false);
        }
    }
}

fn report(t: *const Tally) void {
    std.debug.print("\n{s:<8} {s:>9} {s:>9}  {s:>13}  {s:>13}  {s}\n", .{ "tensor", "compared", "equal", "layer 0 GDN", "layer 3 attn", "largest abs diff" });
    for (kinds) |kind| {
        const k = @intFromEnum(kind);
        var total: usize = 0;
        var equal: usize = 0;
        for (0..41) |l| {
            total += t.total[k][l];
            equal += t.equal[k][l];
        }
        if (total == 0) continue;
        std.debug.print("{s:<8} {d:>9} {d:>9}  {d:>6}/{d:<6}  {d:>6}/{d:<6}  {e}\n", .{ @tagName(kind), total, equal, t.equal[k][0], t.total[k][0], t.equal[k][3], t.total[k][3], t.worst[k] });
    }
}
