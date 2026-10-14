//! A Qwen3.6 MoE layer on CUDA as moe.run lays it out: router, picks, pairs by expert, SwiGLU up and down, combine.

const std = @import("std");
const cuda = @import("cuda");
const rt = @import("router.zig");
const experts = @import("experts.zig");
const shape = @import("qwen27").config;

const A4 = cuda.experts.Affine4Experts;
const prompt_tile = 64; // pairs an item holds in a prompt pass (experts.promptTile)
const decode_tile = 16;

/// The kernels and scratch every MoE layer of a model shares: one layer at a time, at most `rows` rows a call.
pub const Block = struct {
    s: cuda.Stream,
    mods: [3]cuda.Module,
    ex: A4,
    plan: cuda.grouped.Router,
    router: rt.Router,
    sh: rt.Shape,
    count: usize, // routed experts plus the shared one
    width: usize, // a routed expert's inner width
    rows: usize,
    bufs: [10]cuda.DeviceBuffer, // logits, pick, wts, members, items, counts, rank, hist, act, y

    /// `replay`: the captured Triton router, kept beside the native kernels for a step that must fall back to it.
    pub fn load(ctx: *const cuda.Context, s: cuda.Stream, g: shape.Geometry, rows: usize, replay: ?*const cuda.aot.Set) !Block {
        const moe = g.moe orelse return error.UnsupportedQwenGeometry;
        var b: Block = undefined;
        b.s = s;
        b.rows = rows;
        b.sh = .{ .d = g.hidden, .experts = moe.experts, .top_k = moe.top_k };
        b.count = moe.experts + 1;
        b.width = moe.width;
        var made: usize = 0;
        errdefer for (b.mods[0..made]) |*m| m.unload();
        for ([_][]const u8{ cuda.kernels.experts, cuda.kernels.experts_prefill, cuda.kernels.experts_pack }, &b.mods) |blob, *m| {
            m.* = try cuda.Module.load(ctx.d, blob);
            made += 1;
        }
        const sms: usize = @intCast(try ctx.attribute(.multiprocessor_count));
        b.ex = try A4.resolve(b.mods[0], b.mods[1], b.mods[2], sms);
        b.plan = try cuda.grouped.Router.resolve(b.mods[0]);
        b.router = .{ .replay = if (replay) |set| .{ .set = set, .s = s } else null, .native = try rt.Native.load(ctx.d, s, try ctx.capability()) };
        const pairs = rows * (moe.top_k + 1);
        const wide = pairs > cuda.grouped.small;
        const sizes = [10]usize{
            rows * b.count * 4,
            pairs * 4,
            pairs * 4,
            pairs * 4,
            cuda.grouped.maxItems(pairs, b.count, decode_tile) * 12,
            8,
            (if (wide) pairs else 1) * 4,
            (if (wide) (pairs + 1023) / 1024 * b.count else 1) * 4,
            pairs * b.width * 2,
            @max(pairs * g.hidden * 2, @min(rows, 16) * (moe.top_k + 1) * g.hidden * 4), // prompt rows bf16, decode rows fp32
        };
        var bought: usize = 0;
        errdefer for (b.bufs[0..bought]) |*x| x.free();
        for (sizes, &b.bufs) |n, *x| {
            x.* = try cuda.DeviceBuffer.alloc(ctx.d, n);
            bought += 1;
        }
        return b;
    }

    pub fn deinit(b: *Block) void {
        for (&b.bufs) |*x| x.free();
        for (&b.mods) |*m| m.unload();
        b.* = undefined;
    }

    /// The last call's router logits [rows, count] fp32, picks [rows, slots] int32 and weights [rows, slots] fp32.
    pub fn logits(b: *const Block) u64 {
        return b.bufs[0].ptr;
    }
    pub fn picks(b: *const Block) u64 {
        return b.bufs[1].ptr;
    }
    pub fn weights(b: *const Block) u64 {
        return b.bufs[2].ptr;
    }

    /// moe.run: normed rows `x` [rows, hidden] bf16 through layer `m` into `out` (`prompt`: a prompt pass's forms).
    pub fn run(b: *Block, m: experts.Moe, x: u64, rows: usize, prompt: bool, out: u64) !void {
        if (rows == 0 or rows > b.rows) return error.TooManyRows;
        const slots = b.sh.top_k + 1;
        const pairs = rows * slots;
        const tile: usize = if (prompt) prompt_tile else decode_tile;
        try b.router.logits(x, m.router, b.logits(), rows, b.sh);
        try b.router.select(b.logits(), b.picks(), b.weights(), rows, b.sh);
        const plan: cuda.grouped.Plan = .{ .members = b.bufs[3].ptr, .items = b.bufs[4].ptr, .counts = b.bufs[5].ptr, .rank = b.bufs[6].ptr, .hist = b.bufs[7].ptr };
        try b.plan.route(b.s, b.picks(), pairs, b.count, tile, plan);
        const items = cuda.grouped.maxItems(pairs, b.count, tile);
        const l: cuda.experts.Layer = .{ .format = .affine4, .up = m.up, .down = m.down, .width = b.width, .dims = b.sh.d, .experts = b.count, .swiglu = true };
        const g: cuda.experts.Grouped = .{ .affine4 = &b.ex };
        const up_in: cuda.experts.Rows = .{ .x = x, .stride = b.sh.d, .slots = slots };
        const down_in: cuda.experts.Rows = .{ .x = b.bufs[8].ptr, .stride = b.width };
        if (prompt) {
            try g.prompt(b.s, .up, up_in, l, plan, b.bufs[8].ptr, items, -1);
            try g.prompt(b.s, .down_bf16, down_in, l, plan, b.bufs[9].ptr, items, -1);
        } else {
            try g.decode(b.s, .up, up_in, l, plan, b.bufs[8].ptr, items, -1);
            try g.decode(b.s, .down_f32, down_in, l, plan, b.bufs[9].ptr, items, -1);
        }
        try b.router.combine(b.bufs[9].ptr, prompt, b.weights(), out, rows, b.sh);
    }
};
