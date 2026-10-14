//! Qwen3.6 MoE (qwen3_5_moe) on CUDA behind the native family interface (native/cuda.zig's registry).

const std = @import("std");
const cuda = @import("cuda");
const q = @import("qwen27");
const pack = @import("pack.zig");
const weights = @import("weights.zig");
const moe = @import("moe.zig");
const rt = @import("router.zig");
const lanes = @import("lanes");
const mtp = @import("mtp.zig");
const mtp_weights = @import("mtp_weights.zig");
const mtp_drafter = @import("mtp_drafter.zig");

pub const model_type = "qwen3_5_moe";
pub const formats: []const []const u8 = &.{"mlx-q4g64"};
pub const default_context: i64 = 32768;
pub const max_segments: u32 = 1;
pub const prompt_rows: u32 = q.engine.prompt_rows;
/// Streams at once: the 27B's round holds this many windows.
pub const max_lanes: u32 = q.state.max_streams;

pub const Options = struct { context: usize, drafts: bool, segments: usize = 1 };

/// What the native server drives: the 27B's, and the engine the prompt cache copies.
pub const Loaded = struct {
    backend: @FieldType(q.native.Loaded, "backend"),
    facts: @FieldType(q.native.Loaded, "facts"),
    rows: u32,
    stream_bytes: usize,
    free_streams: u32 = 0,
    learn_bytes: usize = 0,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?q.native.LoneRun = null,
    max_lanes: u32,
    target: *anyopaque,
};

/// The prompt cache's copies of a stream's state (qwen27's reuse.zig), the cuts it keeps at, and the rows it needs.
pub const snap_bytes = q.reuse.snapBytes;
pub const snap_save = q.reuse.snapSave;
pub const snap_restore = q.reuse.snapRestore;
pub const snap_drop = q.reuse.snapDrop;
// no draft head reads a row ahead, and the prompt's rows give the same bits cut anywhere: any length keeps, anywhere
pub const snap_rules = .{ .lookahead = 0, .min_prompt = 0 };

/// What this engine refuses before any device work, in words; null when the flags may open it.
pub fn refusal(_: bool, lanes_asked: u32, lanes_fixed: bool) ?[]const u8 {
    if (lanes_fixed and lanes_asked > max_lanes) return std.fmt.comptimePrint("Qwen3.6 MoE on the native CUDA engine serves at most {d} streams at once: serve with --parallel {d} or fewer", .{ max_lanes, max_lanes });
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

/// The engine, its lane backend, the prompt cache's target, and with drafts the MTP head, drafter and wrapper.
const Owned = struct {
    gpa: std.mem.Allocator,
    e: *q.Engine,
    lanes: q.Lanes,
    target: q.reuse.Target,
    head: mtp.Head = undefined,
    drafter: mtp_drafter.Drafter = undefined,
    drafted: lanes.drafted.Drafted = undefined,
    drafting: bool = false,
};

/// Whether every launch of this geometry has a built-in kernel (the dispatch's own shape guards).
fn builtinCovers(g: q.config.Geometry) bool {
    const moe_ = g.moe orelse return false;
    const attn: q.triton.Attn = .{ .heads = g.query_heads, .kv_heads = g.kv_heads, .dim = q.config.head_dim, .half = q.config.rotary_dim / 2 };
    return g.hidden == q.glue_native.hidden and g.linear_v_heads == q.glue_native.value_heads and g.query_heads == q.glue_native.query_heads and
        q.config.linear_dim == q.glue_native.value_dim and q.config.head_dim == q.glue_native.head_dim and
        q.prep_native.gdnFits(g.linear_k_heads, g.linear_v_heads, q.config.linear_dim, g.convDim(), q.config.conv_taps - 1) and
        q.prep_native.attnFits(attn.heads, attn.kv_heads, attn.dim, attn.half) and q.tree_native.fits(attn) and
        g.hidden % rt.stage_columns == 0 and moe_.experts <= 256 and moe_.top_k <= 8;
}

/// The engine for a Qwen3.6 checkpoint: the 27B's, loaded with the MoE weights, hooked with the MoE block.
pub fn engine(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, context: usize) !*q.Engine {
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
    if (kernels == null and !builtinCovers(g)) return error.QwenNeedsCapturedKernels;
    e.k = try q.kernels.Kernels.load(gpa, io, ctx, kernels, g.linear_v_heads);
    errdefer e.k.deinit();
    e.tree = try q.engine.Engine.treeKernels(ctx);
    errdefer if (e.tree) |*n| n.deinit();
    m.block = try moe.Block.load(ctx, e.s, g, q.engine.prompt_rows, if (e.k.triton) |*set| set else null);
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
    return openWith(gpa, io, ctx, dir, kernels, o, null);
}

/// `open`, the MTP head read from `side` when given (a fine-tune of the same language model ships none).
pub fn openWith(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options, side: ?[]const u8) !Loaded {
    // a GB10 runs the built-in kernels alone; a captured set (optional there) is what any other chip needs
    if (kernels == null and (try ctx.capability()) != q.glue_native.verified_capability) return error.QwenNeedsCapturedKernels;
    const e = try engine(gpa, io, ctx, dir, kernels, o.context);
    errdefer e.deinit();
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .e = e, .lanes = try q.Lanes.init(gpa, e, max_lanes), .target = undefined };
    // every prompt through prefill's chunks, the pass checked against the Python engine
    own.lanes.shared_prompts = false;
    errdefer own.lanes.deinit();
    own.target = .{ .e = e, .seqs = &own.lanes.seqs, .held = &own.lanes.held };
    var loaded: Loaded = .{
        .backend = own.lanes.backend(),
        .facts = own.lanes.facts(),
        .rows = q.state.max_rows,
        .stream_bytes = q.state.Seq.bytes(e.w.g, o.context),
        .max_lanes = max_lanes,
        .target = &own.target,
        .ctx = own,
        .deinit = release,
    };
    if (o.drafts) {
        try draftsOn(own, io, ctx, dir, side);
        loaded.backend = own.drafted.backend();
        loaded.facts = own.drafted.facts(own.lanes.facts());
        loaded.stream_bytes = own.drafted.streamBytes(loaded.stream_bytes);
    }
    return loaded;
}

/// The MTP head from the checkpoint's side file (or its own tensors), its drafter and the wrapper.
fn draftsOn(own: *Owned, io: std.Io, ctx: *const cuda.Context, dir: []const u8, side: ?[]const u8) !void {
    const e = own.e;
    const block: *moe.Block = &@as(*Moe, @ptrCast(@alignCast(e.hooks.?.ctx))).block;
    var packk = try pack.Kernels.load(ctx.d, e.s, (try ctx.attribute(.integrated)) == 0);
    defer packk.deinit();
    const w = try mtp_weights.loadFrom(own.gpa, io, &packk, dir, e.w.g, side);
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

test "more lanes than a round holds are refused in words, drafts and fewer are not" {
    try std.testing.expect(refusal(true, 1, true) == null);
    try std.testing.expect(refusal(false, max_lanes + 1, true) != null);
    try std.testing.expect(refusal(false, 8, true) == null);
    try std.testing.expect(refusal(false, 8, false) == null);
    try std.testing.expect(refusal(false, 1, true) == null);
}

test "the built-in kernels cover Qwen3.6's geometry and refuse another" {
    var g: q.config.Geometry = .{ .hidden = 2048, .intermediate = 0, .layers = 40, .linear_k_heads = 16, .linear_v_heads = 32, .query_heads = 16, .kv_heads = 2, .moe = .{ .experts = 256, .top_k = 8, .width = 512, .shared_width = 512 } };
    try std.testing.expect(builtinCovers(g));
    g.hidden = 4096;
    try std.testing.expect(!builtinCovers(g));
    g.hidden = 2048;
    g.kv_heads = 4;
    try std.testing.expect(!builtinCovers(g));
    g.kv_heads = 2;
    g.moe = null;
    try std.testing.expect(!builtinCovers(g));
}
