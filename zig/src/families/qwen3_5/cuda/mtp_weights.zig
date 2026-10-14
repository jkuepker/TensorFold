//! The Qwen3.6 MTP layer on the GPU as the Python load_mtp packs it: fc, norms, attention, experts, draft head.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const qwen35 = @import("qwen35");
const w27 = @import("qwen27").weights;
const shape = @import("qwen27").config;
const pack = @import("pack.zig");
const experts = @import("experts.zig");
const draft_ids = @import("draft_ids.zig");
const Named = @import("weights.zig").Named;

const QLinear = w27.QLinear;
const Tensor = qwen35.host.Tensor;

pub const Head = struct {
    /// The device buffers this head owns (its geometry is the trunk's).
    base: w27.Weights,
    norm_e: u64,
    norm_h: u64,
    fc_e: QLinear,
    fc_h: QLinear,
    input_norm: u64,
    post_norm: u64,
    attn: w27.Attention,
    moe: experts.Moe,
    norm: u64,
    /// The draft head: column j scores token `ids[j]`.
    draft: QLinear,
    ids: []u32,
    named: std.ArrayList(Named) = .empty,

    pub fn deinit(h: *Head) void {
        const gpa = h.base.gpa;
        for (h.named.items) |n| gpa.free(n.name);
        h.named.deinit(gpa);
        gpa.free(h.ids);
        h.base.deinit();
        h.* = undefined;
    }

    pub fn bytes(h: *const Head) usize {
        return h.base.bytes();
    }

    fn note(h: *Head, ptr: u64, len: usize, name: []const u8) !void {
        const copy = try h.base.gpa.dupe(u8, name);
        errdefer h.base.gpa.free(copy);
        try h.named.append(h.base.gpa, .{ .name = copy, .ptr = ptr, .len = len });
    }

    fn linear(h: *Head, name: []const u8, q: QLinear) !void {
        const sizes = cuda.qlinear.Affine4.layout(q.n, q.k);
        var buf: [64]u8 = undefined;
        for ([_]struct { []const u8, u64, usize }{ .{ "weight", q.w, sizes.words }, .{ "scales", q.s, sizes.scales }, .{ "biases", q.b, sizes.scales } }) |part| {
            try h.note(part[1], part[2], try std.fmt.bufPrint(&buf, "{s}.{s}", .{ name, part[0] }));
        }
    }

    /// Every tensor by the names of the oracle's mtp_weights.txt.
    fn names(h: *Head, g: shape.Geometry) !void {
        try h.note(h.norm_e, g.hidden * 2, "mtp.norm_e");
        try h.note(h.norm_h, g.hidden * 2, "mtp.norm_h");
        try h.linear("mtp.fc_e", h.fc_e);
        try h.linear("mtp.fc_h", h.fc_h);
        try h.note(h.input_norm, g.hidden * 2, "mtp.input_norm");
        try h.note(h.post_norm, g.hidden * 2, "mtp.post_norm");
        try h.linear("mtp.attn.q", h.attn.q);
        try h.linear("mtp.attn.k", h.attn.k);
        try h.linear("mtp.attn.v", h.attn.v);
        try h.linear("mtp.attn.o", h.attn.o);
        try h.note(h.attn.q_norm, shape.head_dim * 2, "mtp.attn.q_norm");
        try h.note(h.attn.k_norm, shape.head_dim * 2, "mtp.attn.k_norm");
        try h.note(h.moe.router, h.moe.router_bytes, "mtp.moe.router");
        try h.note(h.moe.up, h.moe.up_bytes, "mtp.moe.up");
        try h.note(h.moe.down, h.moe.down_bytes, "mtp.moe.down");
        try h.note(h.norm, g.hidden * 2, "mtp.norm");
        try h.linear("mtp.head", h.draft);
    }
};

/// A norm weight widened to fp32 by the host loader, back to the bf16 it was stored as (exact, or refused).
fn bf16Vector(L: *w27.Loader, t: Tensor, len: usize) !u64 {
    if (t.dtype != .f32 or t.bytes.len != len * 4) return error.UnexpectedTensor;
    const out = std.mem.bytesAsSlice(u16, try L.staging(len * 2));
    for (out, 0..) |*o, i| {
        const w = std.mem.readInt(u32, t.bytes[4 * i ..][0..4], .little);
        if (w & 0xffff != 0) return error.UnexpectedTensor;
        o.* = @intCast(w >> 16);
    }
    const ptr = try L.alloc(len * 2);
    try L.pack.upload(ptr, L.host.items);
    return ptr;
}

/// A host projection's tensors tiled onto the device: 4-bit group-64 MLX words, scales and biases.
fn tiled(L: *w27.Loader, name: []const u8, p: qwen35.host.Projection, n: usize, k: usize) !QLinear {
    const x = switch (p) {
        .mlx => |m| m,
        else => return error.UnsupportedQuantization,
    };
    if (x.bits != 4 or x.group != 64) return error.UnsupportedQuantization;
    return L.tileTensors(.{ x.words, x.scales, x.biases }, name, n, k);
}

/// The draft head: `ids` rows of the trunk's lm_head, gathered on the host, then tiled.
fn draftHead(gpa: std.mem.Allocator, L: *w27.Loader, ck: *core.Checkpoint, ids: []const u32, g: shape.Geometry) !QLinear {
    const parts = [3]struct { name: []const u8, width: usize }{
        .{ .name = w27.prefix ++ "lm_head.weight", .width = g.hidden / 8 * 4 },
        .{ .name = w27.prefix ++ "lm_head.scales", .width = g.hidden / 64 * 2 },
        .{ .name = w27.prefix ++ "lm_head.biases", .width = g.hidden / 64 * 2 },
    };
    var gathered: [3]Tensor = undefined;
    var made: usize = 0;
    defer for (gathered[0..made]) |t| gpa.free(@constCast(t.bytes));
    for (parts, &gathered) |p, *out| {
        const t = try ck.get(p.name);
        if (t.rank != 2 or t.shape[0] != g.vocab or t.bytes.len != g.vocab * p.width) return error.UnexpectedTensor;
        const rows = try gpa.alloc(u8, ids.len * p.width);
        for (ids, 0..) |id, j| @memcpy(rows[j * p.width ..][0..p.width], t.bytes[@as(usize, id) * p.width ..][0..p.width]);
        out.* = t;
        out.shape[0] = ids.len;
        out.bytes = rows;
        made += 1;
    }
    const q = try L.tileTensors(gathered, "mtp.head", ids.len, g.hidden);
    try L.settle(); // the gathered rows go when this returns
    return q;
}

/// The MTP layer beside the checkpoint (its side file, else its own shards) on the GPU, the trunk's geometry `g`.
pub fn load(gpa: std.mem.Allocator, io: std.Io, k: *const pack.Kernels, dir: []const u8, g: shape.Geometry) !Head {
    return loadFrom(gpa, io, k, dir, g, null);
}

/// `load` reading the MTP layer from `side` when given (Qwen3.6's file beside a fine-tune that ships none).
pub fn loadFrom(gpa: std.mem.Allocator, io: std.Io, k: *const pack.Kernels, dir: []const u8, g: shape.Geometry, side: ?[]const u8) !Head {
    var host = try qwen35.Checkpoint.open(gpa, io, dir);
    defer host.close();
    if (side) |path| try host.useSide(path);
    const moe = g.moe orelse return error.UnsupportedQwenGeometry;
    var m = (try host.mtp()) orelse return error.QwenCheckpointHasNoDraftHead;
    defer m.deinit();
    const routed = switch (m.mlp orelse return error.QwenCheckpointHasNoDraftHead) {
        .routed => |r| r,
        .dense => return error.UnsupportedQwenGeometry,
    };
    if (!m.gated or m.head != null) return error.UnsupportedQwenGeometry;
    var h: Head = undefined;
    h.base = .{ .gpa = gpa, .g = g };
    h.named = .empty;
    h.ids = try draft_ids.load(gpa, g.vocab);
    errdefer gpa.free(h.ids);
    errdefer h.base.deinit();
    var src = try w27.Source.init(gpa, .{ .d = k.pack.d, .s = k.pack.s, .discrete = k.pack.discrete });
    defer src.deinit();
    var L = try w27.open(gpa, k.pack, &h.base, &src);
    defer L.close();
    h.norm_e = try bf16Vector(&L, m.fc_e_norm, g.hidden);
    h.norm_h = try bf16Vector(&L, m.fc_h_norm, g.hidden);
    h.fc_e = try tiled(&L, "mtp.fc_e", m.fc_e, g.hidden, g.hidden);
    h.fc_h = try tiled(&L, "mtp.fc_h", m.fc_h, g.hidden, g.hidden);
    h.input_norm = try bf16Vector(&L, m.input_norm.?, g.hidden);
    h.post_norm = try bf16Vector(&L, m.post_norm.?, g.hidden);
    h.attn = .{
        .q = try tiled(&L, "mtp.attn.q", m.q, 2 * g.qInner(), g.hidden),
        .k = try tiled(&L, "mtp.attn.k", m.k, g.kvInner(), g.hidden),
        .v = try tiled(&L, "mtp.attn.v", m.v, g.kvInner(), g.hidden),
        .o = try tiled(&L, "mtp.attn.o", m.o, g.hidden, g.qInner()),
        .q_norm = try bf16Vector(&L, m.q_norm, shape.head_dim),
        .k_norm = try bf16Vector(&L, m.k_norm, shape.head_dim),
    };
    h.moe = try experts.upload(&L, k, routed, g, moe);
    h.norm = try bf16Vector(&L, m.final_norm, g.hidden);
    try L.settle();
    var ck = try core.Checkpoint.openModelSkipping(gpa, io, dir, w27.prefix, "vision");
    defer ck.close();
    h.draft = try draftHead(gpa, &L, &ck, h.ids, g);
    try L.settle();
    try h.names(g);
    return h;
}
