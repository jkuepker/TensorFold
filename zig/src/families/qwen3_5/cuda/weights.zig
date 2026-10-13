//! Qwen3.5-family MoE weights on the GPU: the 27B's tiled projections, norms and DeltaNet tensors, routed experts last.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const qwen35 = @import("qwen35");
const w27 = @import("qwen27").weights;
const shape = @import("qwen27").config;
const pack = @import("pack.zig");
const experts = @import("experts.zig");

pub const Moe = experts.Moe;
pub const QLinear = w27.QLinear;
pub const Layer = struct { input_norm: u64, post_norm: u64, mixer: w27.Mixer, moe: Moe };

/// A device tensor by the Python engine's names (weights.txt of the oracle), for digests against its weights.
pub const Named = struct { name: []u8, ptr: u64, len: usize };

pub const Weights = struct {
    /// The embedding, final norm, head, rotary table, end ids and every device buffer.
    base: w27.Weights,
    layers: []Layer = &.{},
    named: std.ArrayList(Named) = .empty,

    pub fn deinit(w: *Weights) void {
        for (w.named.items) |n| w.base.gpa.free(n.name);
        w.named.deinit(w.base.gpa);
        w.base.gpa.free(w.layers);
        w.base.deinit();
        w.* = undefined;
    }

    /// Device bytes the weights hold.
    pub fn bytes(w: *const Weights) usize {
        return w.base.bytes();
    }

    fn note(w: *Weights, ptr: u64, len: usize, comptime fmt: []const u8, args: anytype) !void {
        const name = try std.fmt.allocPrint(w.base.gpa, fmt, args);
        errdefer w.base.gpa.free(name);
        try w.named.append(w.base.gpa, .{ .name = name, .ptr = ptr, .len = len });
    }

    fn linear(w: *Weights, comptime fmt: []const u8, args: anytype, q: QLinear) !void {
        const sizes = cuda.qlinear.Affine4.layout(q.n, q.k);
        inline for (.{ .{ "weight", q.w, sizes.words }, .{ "scales", q.s, sizes.scales }, .{ "biases", q.b, sizes.scales } }) |part| {
            try w.note(part[1], part[2], fmt ++ ".{s}", args ++ .{part[0]});
        }
    }

    /// Every tensor of layer `i` and the ends, named as the oracle's weights.txt names them.
    fn names(w: *Weights, g: shape.Geometry) !void {
        const embed = w.base.embed;
        try w.note(embed.w, g.vocab * g.hidden / 2, "embed.weight", .{});
        try w.note(embed.s, g.vocab * g.hidden / 64 * 2, "embed.scales", .{});
        try w.note(embed.b, g.vocab * g.hidden / 64 * 2, "embed.biases", .{});
        try w.note(w.base.norm, g.hidden * 2, "norm", .{});
        try w.linear("head", .{}, w.base.head);
        try w.note(w.base.inv_freq, shape.rotary_dim / 2 * 4, "inv_freq", .{});
        for (w.layers, 0..) |l, i| {
            try w.note(l.input_norm, g.hidden * 2, "L{d:0>2}.input_norm", .{i});
            try w.note(l.post_norm, g.hidden * 2, "L{d:0>2}.post_norm", .{i});
            switch (l.mixer) {
                .delta => |d| {
                    inline for (.{ "qkv", "z", "b", "a", "out" }) |n| try w.linear("L{d:0>2}.gdn." ++ n, .{i}, @field(d, n));
                    try w.note(d.conv, g.convDim() * shape.conv_taps * 2, "L{d:0>2}.gdn.conv", .{i});
                    try w.note(d.a_log, g.linear_v_heads * 4, "L{d:0>2}.gdn.A_log", .{i});
                    try w.note(d.dt_bias, g.linear_v_heads * 4, "L{d:0>2}.gdn.dt_bias", .{i});
                    try w.note(d.norm, shape.linear_dim * 2, "L{d:0>2}.gdn.norm", .{i});
                },
                .attention => |a| {
                    inline for (.{ "q", "k", "v", "o" }) |n| try w.linear("L{d:0>2}.attn." ++ n, .{i}, @field(a, n));
                    try w.note(a.q_norm, shape.head_dim * 2, "L{d:0>2}.attn.q_norm", .{i});
                    try w.note(a.k_norm, shape.head_dim * 2, "L{d:0>2}.attn.k_norm", .{i});
                },
            }
            try w.note(l.moe.router, l.moe.router_bytes, "L{d:0>2}.moe.router", .{i});
            try w.note(l.moe.up, l.moe.up_bytes, "L{d:0>2}.moe.up", .{i});
            try w.note(l.moe.down, l.moe.down_bytes, "L{d:0>2}.moe.down", .{i});
        }
    }
};

/// The model folder's text tensors on the GPU: trunk tensors as the 27B's loader tiles them, each layer's router and
/// experts from the host loader (8-bit routers widened to bf16 there), leftovers refused.
pub fn load(gpa: std.mem.Allocator, io: std.Io, k: *const pack.Kernels, dir: []const u8) !Weights {
    var host = try qwen35.Checkpoint.open(gpa, io, dir);
    defer host.close();
    const cfg = try shape.fromSpec(host.spec(), host.config.eos);
    const g = cfg.g;
    const moe = g.moe orelse return error.UnsupportedQwenGeometry; // a dense checkpoint is the 27B engine's own
    if (host.config.tied) return error.UnsupportedQwenGeometry;
    var w: Weights = .{ .base = .{ .gpa = gpa, .g = g, .eos = cfg.eos, .eos_count = cfg.eos_count } };
    errdefer w.deinit();
    var src = try w27.Source.init(gpa, .{ .d = k.pack.d, .s = k.pack.s, .discrete = k.pack.discrete });
    defer src.deinit();
    var L = try w27.open(gpa, k.pack, &w.base, &src);
    defer L.close();
    var ck = try core.Checkpoint.openModelPrefix(gpa, io, dir, w27.prefix);
    defer ck.close();
    try src.add(&ck);
    w.layers = try gpa.alloc(Layer, g.layers);
    @memset(w.layers, undefined);
    for (w.layers, 0..) |*layer, i| {
        var pb: [64]u8 = undefined;
        var nb: [160]u8 = undefined;
        const p = try std.fmt.bufPrint(&pb, "model.layers.{d}.", .{i});
        layer.input_norm = try L.vector(&ck, try std.fmt.bufPrint(&nb, "{s}input_layernorm.weight", .{p}), g.hidden);
        layer.mixer = if (shape.linear(i)) .{ .delta = try L.delta(&ck, p, g) } else .{ .attention = try L.attention(&ck, p, g) };
        layer.post_norm = try L.vector(&ck, try std.fmt.bufPrint(&nb, "{s}post_attention_layernorm.weight", .{p}), g.hidden);
        // the MLP's tensors come through the host loader: the trunk's checkpoint leaves them alone
        try ck.skip(try std.fmt.bufPrint(&nb, "{s}{s}mlp.", .{ w27.prefix, p }));
        var layer_host = try host.layer(i);
        defer layer_host.deinit();
        const mlp = switch (layer_host.body) {
            inline else => |b| b.mlp,
        };
        const routed = switch (mlp) {
            .routed => |r| r,
            .dense => return error.UnsupportedQwenGeometry,
        };
        layer.moe = try experts.upload(&L, k, routed, g, moe);
        try L.settle(); // the layer's host memory goes next
    }
    try L.ends(&ck, g.hidden);
    if (ck.unused() != 0) return error.UnusedCheckpointTensors;
    try L.rotary();
    try L.settle();
    try w.names(g);
    // the 27B's forward walks base.layers: these mixers and norms, its dense MLP unused under the MoE hook
    w.base.layers = try gpa.alloc(w27.Layer, g.layers);
    for (w.base.layers, w.layers) |*b, l| b.* = .{ .input_norm = l.input_norm, .post_norm = l.post_norm, .mixer = l.mixer, .gate = undefined, .up = undefined, .down = undefined };
    return w;
}
