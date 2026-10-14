//! The two fatbins the weight upload runs and nothing else: affine4_pack's tiles and experts_pack's blocks.

const std = @import("std");
const cuda = @import("cuda");
const w27 = @import("qwen27").weights;

fn int(x: usize) c_int {
    return @intCast(x);
}

pub const Kernels = struct {
    mods: [2]cuda.Module,
    experts_pack: cuda.Function,
    /// The 4-bit tiling functions and stream the 27B's loader helpers run on.
    pack: w27.Pack,

    pub fn load(d: *const cuda.Driver, s: cuda.Stream, discrete: bool) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = undefined;
        k.mods[0] = try cuda.Module.load(d, cuda.kernels.affine4_pack);
        errdefer k.mods[0].unload();
        k.mods[1] = try cuda.Module.load(d, cuda.kernels.experts_pack);
        errdefer k.mods[1].unload();
        k.experts_pack = try k.mods[1].function(cuda.experts.Affine4Experts.symbols.pack);
        k.pack = .{
            .d = d,
            .s = s,
            .pack_dense = try k.mods[0].function(cuda.qlinear.symbols.pack_words),
            .transpose16 = try k.mods[0].function(cuda.qlinear.symbols.pack_scales),
            .discrete = discrete,
        };
        return k;
    }

    pub fn deinit(k: *Kernels) void {
        for (&k.mods) |*m| m.unload();
    }

    /// experts.make for one matrix: `count` experts' MLX words, scales and biases into `out`'s blocks.
    pub fn experts(k: *const Kernels, words: u64, scales: u64, biases: u64, out: u64, count: usize, n: usize, kk: usize) !void {
        var a: cuda.Args = .{};
        for ([_]u64{ words, scales, biases, out }) |v| a.add(v);
        for ([_]usize{ n, kk / 8, kk / 64, n / 32 }) |v| a.add(int(v));
        const cfg: cuda.Config = .{ .grid = .{ .x = @intCast(kk / 64), .y = @intCast(n / 32), .z = @intCast(count) }, .block = .{ .x = 288 } };
        try cuda.launch.launch(k.experts_pack, cfg, k.pack.s, &a);
    }
};
