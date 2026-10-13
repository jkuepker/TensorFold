//! Qwen3.6 MoE (qwen3_5_moe) on CUDA behind the native family interface (native/cuda.zig's registry): the 27B's engine with the MoE block in place of its dense MLP.

const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen27");
const pack = @import("pack.zig");
const weights = @import("weights.zig");
const moe = @import("moe.zig");
const lanes = @import("lanes");
const mtp = @import("mtp.zig");
const mtp_weights = @import("mtp_weights.zig");
const mtp_drafter = @import("mtp_drafter.zig");

pub const model_type = "qwen3_5_moe";
pub const formats: []const []const u8 = &.{"mlx-q4g64"};
pub const default_context: i64 = 32768;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = q.engine.prompt_rows;
/// Streams at once: serial only, until the MoE rows batch across lanes.
pub const max_lanes: u32 = 1;

pub const Options = struct { context: usize, drafts: bool, segments: usize = 1 };
pub const Loaded = q.native.Loaded;

/// What this engine refuses before any device work, in words; null when the flags may open it.
pub fn refusal(_: bool, lanes_asked: u32, lanes_fixed: bool) ?[]const u8 {
    if (lanes_fixed and lanes_asked > max_lanes) return "Qwen3.6 MoE serves one stream at a time on the native CUDA engine so far: serve with --parallel 1";
    return null;
}

/// The checkpoint's MoE weights and block, owned by the engine in place of its dense weights.
pub const Moe = struct {
    gpa: std.mem.Allocator,
    w: weights.Weights,
    block: moe.Block,

    fn mlp(p: *anyopaque, layer: usize, y: u64, _: u64, rows: usize, prompt: bool, out: u64) anyerror!void {
        const m: *Moe = @ptrCast(@alignCast(p));
        try m.block.run(m.w.layers[layer].moe, y, rows, prompt, out);
    }

    fn deinit(p: *anyopaque) void {
        const m: *Moe = @ptrCast(@alignCast(p));
        m.block.deinit();
        m.w.deinit();
        m.gpa.destroy(m);
    }
};

/// The engine, its lane backend, and with drafts the MTP head, its drafter and the wrapper that joins them.
const Owned = struct {
    gpa: std.mem.Allocator,
    e: *q.Engine,
    lanes: q.Lanes,
    head: mtp.Head = undefined,
    drafter: mtp_drafter.Drafter = undefined,
    drafted: lanes.drafted.Drafted = undefined,
    drafting: bool = false,
};

/// The engine for a Qwen3.6 checkpoint: the 27B's, loaded with the MoE weights, hooked with the MoE block.
pub fn engine(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: []const u8, context: usize) !*q.Engine {
    const e = try gpa.create(q.Engine);
    errdefer gpa.destroy(e);
    e.gpa = gpa;
    e.ctx = ctx;
    e.context = context;
    e.hooks = null;
    e.owner = null;
    e.s = try cuda.Stream.init(ctx.d, true);
    errdefer e.s.deinit();
    const m = try gpa.create(Moe);
    errdefer gpa.destroy(m);
    m.gpa = gpa;
    {
        var packk = try pack.Kernels.load(ctx.d, e.s, (try ctx.attribute(.integrated)) == 0);
        defer packk.deinit();
        m.w = try weights.load(gpa, io, &packk, dir);
        try e.s.synchronize();
    }
    errdefer m.w.deinit();
    const g = m.w.base.g;
    e.k = try q.kernels.Kernels.load(gpa, io, ctx, kernels, g.linear_v_heads);
    errdefer e.k.deinit();
    m.block = try moe.Block.load(ctx, e.s, g, q.engine.prompt_rows, &e.k.triton);
    errdefer m.block.deinit();
    e.w = m.w.base;
    e.scratch = try q.state.Scratch.init(gpa, ctx.d, g, q.engine.prompt_rows, context);
    errdefer e.scratch.deinit();
    e.hooks = .{ .ctx = m, .mlp = Moe.mlp };
    e.owner = .{ .ptr = m, .deinit = Moe.deinit };
    try e.s.synchronize();
    return e;
}

pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    // the glue runs the Python engine's Triton kernels, captured on this chip (no own glue kernels yet)
    const e = try engine(gpa, io, ctx, dir, kernels orelse return error.QwenNeedsCapturedKernels, o.context);
    errdefer e.deinit();
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .e = e, .lanes = try q.Lanes.init(gpa, e, max_lanes) };
    own.lanes.shared_prompts = false; // every prompt through prefill's chunks, the pass checked against the Python engine
    errdefer own.lanes.deinit();
    var loaded: Loaded = .{
        .backend = own.lanes.backend(),
        .facts = own.lanes.facts(),
        .rows = q.state.max_rows,
        .stream_bytes = q.state.Seq.bytes(e.w.g, o.context),
        .max_lanes = max_lanes,
        .ctx = own,
        .deinit = release,
    };
    if (o.drafts) {
        try draftsOn(own, io, ctx, dir);
        loaded.backend = own.drafted.backend();
        loaded.facts = own.drafted.facts(own.lanes.facts());
        loaded.stream_bytes = own.drafted.streamBytes(loaded.stream_bytes);
    }
    return loaded;
}

/// The MTP head from the checkpoint's side file (or its own tensors), its drafter and the wrapper; window costs timed first.
fn draftsOn(own: *Owned, io: std.Io, ctx: *const cuda.Context, dir: []const u8) !void {
    const e = own.e;
    const block: *moe.Block = &@as(*Moe, @ptrCast(@alignCast(e.hooks.?.ctx))).block;
    var packk = try pack.Kernels.load(ctx.d, e.s, (try ctx.attribute(.integrated)) == 0);
    defer packk.deinit();
    const w = try mtp_weights.load(own.gpa, io, &packk, dir, e.w.g);
    own.head = try mtp.Head.init(own.gpa, e, w, block);
    errdefer own.head.deinit();
    try e.s.synchronize();
    try own.lanes.calibrate(io, 1);
    own.drafter = try mtp_drafter.Drafter.init(own.gpa, io, &own.head);
    errdefer own.drafter.deinit();
    own.drafted = try lanes.drafted.Drafted.init(own.gpa, own.lanes.backend(), own.drafter.drafter());
    own.drafting = true;
    std.log.info("Qwen3.6 MTP drafts: {d} a round, {d} draft ids, head {d} MiB", .{ mtp_drafter.depth, own.head.w.ids.len, own.head.bytes() >> 20 });
}

/// A request this engine refuses, in words; null: none of its own.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PromptTooLong => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.OutOfDeviceMemory => "the GPU had no memory left for this request's caches: retry once another request ends",
        error.QwenCheckpointHasNoDraftHead => "this checkpoint carries no MTP head for the native CUDA engine: serve with --no-drafts",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    if (own.drafting) {
        own.drafted.deinit();
        own.drafter.deinit();
        own.head.deinit();
    }
    own.lanes.deinit();
    own.e.deinit();
    own.gpa.destroy(own);
}

test "extra lanes are refused in words, drafts and one stream are not" {
    try std.testing.expect(refusal(true, 1, true) == null);
    try std.testing.expect(refusal(false, 2, true) != null);
    try std.testing.expect(refusal(false, 8, false) == null);
    try std.testing.expect(refusal(false, 1, true) == null);
}
