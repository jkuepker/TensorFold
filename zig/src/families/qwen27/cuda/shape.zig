//! The shapes the CUDA engine is built for: the family's parsed config (config.zig), held to the 27B's.
const std = @import("std");
const config = @import("../config.zig");

pub const vocab = 248320;
pub const linear_dim = 128; // a DeltaNet key or value head's width
pub const conv_taps = 4;
pub const head_dim = 256;
pub const rotary_dim = 64;
pub const eps: f32 = 1e-6;
pub const theta: f32 = 10000000;

/// Whether layer `index` is a DeltaNet layer: every fourth is attention.
pub fn linear(index: usize) bool {
    return index % 4 != 3;
}

/// The routed MLP a MoE layer holds instead of the dense gate, up and down; the shared expert is expert `experts`.
pub const Moe = struct {
    experts: usize,
    top_k: usize,
    width: usize, // a routed expert's inner width
    shared_width: usize,
};

/// The widths a forward sizes its buffers by.
pub const Geometry = struct {
    hidden: usize,
    intermediate: usize, // the dense MLP's inner width, 0 with `moe`
    layers: usize,
    linear_k_heads: usize, // DeltaNet key (and query) heads
    linear_v_heads: usize, // DeltaNet value heads, each key head serving a run of them
    query_heads: usize,
    kv_heads: usize,
    moe: ?Moe = null,

    /// The DeltaNet input projection's row: queries, keys, then values.
    pub fn convDim(g: Geometry) usize {
        return (2 * g.linear_k_heads + g.linear_v_heads) * linear_dim;
    }
    pub fn kInner(g: Geometry) usize {
        return g.linear_k_heads * linear_dim;
    }
    pub fn vInner(g: Geometry) usize {
        return g.linear_v_heads * linear_dim;
    }
    pub fn qInner(g: Geometry) usize {
        return g.query_heads * head_dim;
    }
    pub fn kvInner(g: Geometry) usize {
        return g.kv_heads * head_dim;
    }
    /// The widest row an activation buffer holds: the recurrence's values or the attention's heads.
    pub fn inner(g: Geometry) usize {
        return @max(g.vInner(), g.qInner());
    }
    /// One layer's recurrent state (fp32) and its conv window (bf16).
    pub fn deltaBytes(g: Geometry) usize {
        return g.linear_v_heads * linear_dim * linear_dim * 4;
    }
    pub fn convBytes(g: Geometry) usize {
        return (conv_taps - 1) * g.convDim() * 2;
    }
};

/// Qwen3.8-27B, the one checkpoint shape the CUDA engine serves.
pub const the_27b: Geometry = .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .linear_k_heads = 16, .linear_v_heads = 48, .query_heads = 24, .kv_heads = 4 };

/// Qwen3.6-35B-A3B: the 27B's layer kinds and head widths over 256 routed experts and a shared one.
pub const the_35b_a3b: Geometry = .{
    .hidden = 2048,
    .intermediate = 0,
    .layers = 40,
    .linear_k_heads = 16,
    .linear_v_heads = 32,
    .query_heads = 16,
    .kv_heads = 2,
    .moe = .{ .experts = 256, .top_k = 8, .width = 512, .shared_width = 512 },
};

/// A checkpoint's config.json (and generation_config.json's end tokens), refused unless it is the 27B's.
pub const Config = struct {
    g: Geometry,
    eos: [config.max_eos]u32,
    eos_count: usize,

    pub fn read(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Config {
        const text = try file(gpa, io, dir, "config.json") orelse return error.MissingConfig;
        defer gpa.free(text);
        const generation = try file(gpa, io, dir, "generation_config.json");
        defer if (generation) |x| gpa.free(x);
        return check(try config.parse(gpa, text, generation));
    }

    fn file(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) !?[]u8 {
        const path = try std.fs.path.join(gpa, &.{ dir, name });
        defer gpa.free(path);
        return std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22)) catch |err| switch (err) {
            error.FileNotFound => null,
            else => err,
        };
    }
};

/// The parsed config as the CUDA engine's Config, when every shape the kernels assume is the 27B's.
pub fn check(c: config.Config) !Config {
    const g: Geometry = .{ .hidden = c.hidden, .intermediate = c.intermediate, .layers = c.layers, .linear_k_heads = c.k_heads, .linear_v_heads = c.v_heads, .query_heads = c.heads, .kv_heads = c.kv_heads };
    if (!std.meta.eql(g, the_27b) or c.vocab != vocab or c.head_dim != head_dim or c.dk != linear_dim or c.dv != linear_dim) return error.UnsupportedQwenGeometry;
    if (c.conv_kernel != conv_taps or c.rope_dims != rotary_dim or c.eps != eps or c.rope_theta != theta) return error.UnsupportedQwenGeometry;
    for (c.kinds[0..c.layers], 0..) |kind, i| if ((kind == .linear) != linear(i)) return error.UnsupportedQwenGeometry;
    return .{ .g = g, .eos = c.eos, .eos_count = c.eos_count };
}

/// The checkpoint shapes the weight upload serves: each pins every width the kernels assume.
pub const known = [_]Geometry{ the_27b, the_35b_a3b };

/// A host loader's spec (qwen3_5/config.zig `Spec`) and end ids as a Config, when it is a known shape.
pub fn fromSpec(spec: anytype, eos: []const u32) !Config {
    const moe: ?Moe = if (spec.experts == 0) null else .{ .experts = spec.experts, .top_k = spec.top_k, .width = spec.moe_width, .shared_width = spec.shared_width };
    const g: Geometry = .{ .hidden = spec.hidden, .intermediate = if (moe == null) spec.intermediate else 0, .layers = spec.n_layers, .linear_k_heads = spec.key_heads, .linear_v_heads = spec.value_heads, .query_heads = spec.heads, .kv_heads = spec.kv_heads, .moe = moe };
    const listed = for (known) |k| {
        if (std.meta.eql(g, k)) break true;
    } else false;
    if (!listed or spec.vocab != vocab or spec.head_dim != head_dim or spec.key_dim != linear_dim or spec.value_dim != linear_dim) return error.UnsupportedQwenGeometry;
    if (spec.conv != conv_taps or spec.rotary_dim != rotary_dim or @as(f32, @floatCast(spec.eps)) != eps or @as(f32, @floatCast(spec.rope_theta)) != theta or spec.bits != 4 or spec.group != 64) return error.UnsupportedQwenGeometry;
    for (0..spec.n_layers) |i| if (spec.full(i) == linear(i)) return error.UnsupportedQwenGeometry;
    if (eos.len > config.max_eos) return error.UnsupportedQwenGeometry;
    var out: Config = .{ .g = g, .eos = undefined, .eos_count = eos.len };
    @memcpy(out.eos[0..eos.len], eos);
    return out;
}

test "the 35B-A3B's spec passes, the 27B's check refuses it, and a changed width is refused" {
    const Spec = struct {
        hidden: usize = 2048,
        intermediate: usize = 0,
        n_layers: usize = 40,
        heads: usize = 16,
        kv_heads: usize = 2,
        head_dim: usize = head_dim,
        key_heads: usize = 16,
        value_heads: usize = 32,
        key_dim: usize = linear_dim,
        value_dim: usize = linear_dim,
        conv: usize = conv_taps,
        vocab: usize = vocab,
        eps: f64 = 1e-6,
        rope_theta: f64 = 1e7,
        rotary_dim: usize = rotary_dim,
        bits: u8 = 4,
        group: u16 = 64,
        experts: usize = 256,
        top_k: usize = 8,
        moe_width: usize = 512,
        shared_width: usize = 512,

        fn full(_: @This(), i: usize) bool {
            return !linear(i);
        }
    };
    var s: Spec = .{};
    const ok = try fromSpec(s, &.{ 248046, 248044 });
    try std.testing.expectEqual(@as(usize, 2), ok.eos_count);
    try std.testing.expectEqual(@as(usize, 8), ok.g.moe.?.top_k);
    s.moe_width = 256;
    try std.testing.expectError(error.UnsupportedQwenGeometry, fromSpec(s, &.{248044}));
    s = .{};
    s.bits = 8;
    try std.testing.expectError(error.UnsupportedQwenGeometry, fromSpec(s, &.{248044}));
    var c: config.Config = .{ .hidden = 2048, .intermediate = 0, .layers = 40, .vocab = vocab, .heads = 16, .kv_heads = 2, .head_dim = head_dim, .k_heads = 16, .v_heads = 32, .dk = linear_dim, .dv = linear_dim, .conv_kernel = conv_taps, .eps = eps, .rope_dims = rotary_dim, .rope_theta = theta };
    for (c.kinds[0..40], 0..) |*k, i| k.* = if (linear(i)) .linear else .attention;
    try std.testing.expectError(error.UnsupportedQwenGeometry, check(c));
}

test "the 27B's shapes pass and another hidden size is refused" {
    var c: config.Config = .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = vocab, .heads = 24, .kv_heads = 4, .head_dim = head_dim, .k_heads = 16, .v_heads = 48, .dk = linear_dim, .dv = linear_dim, .conv_kernel = conv_taps, .eps = eps, .rope_dims = rotary_dim, .rope_theta = theta };
    for (c.kinds[0..64], 0..) |*k, i| k.* = if (linear(i)) .linear else .attention;
    c.eos[0] = 248044;
    c.eos_count = 1;
    const ok = try check(c);
    try std.testing.expectEqual(@as(usize, 1), ok.eos_count);
    c.hidden = 2048;
    try std.testing.expectError(error.UnsupportedQwenGeometry, check(c));
}
