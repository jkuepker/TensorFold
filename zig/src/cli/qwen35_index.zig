//! tf-qwen35-index DIR: opens a Qwen3.5 / Qwen3.6 checkpoint through the host loader and prints its config and tensor index.
const std = @import("std");
const qwen = @import("qwen35");

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) {
        std.debug.print("usage: tf-qwen35-index CHECKPOINT_DIR\n", .{});
        return 2;
    }
    var ck = try qwen.Checkpoint.open(gpa, init.io, args[1]);
    defer ck.close();
    const s = ck.spec();
    std.debug.print("config: hidden {d}, layers {d}, heads {d}/{d} head_dim {d}, rotary {d}, theta {d}, vocab {d}, eps {e}\n", .{ s.hidden, s.n_layers, s.heads, s.kv_heads, s.head_dim, s.rotary_dim, s.rope_theta, s.vocab, s.eps });
    std.debug.print("config: delta-net key {d}x{d}, value {d}x{d}, conv {d}; experts {d} top-{d}, width {d}, shared {d}; mtp layers {d}\n", .{ s.key_heads, s.key_dim, s.value_heads, s.value_dim, s.conv, s.experts, s.top_k, s.moe_width, s.shared_width, s.mtp_layers });
    std.debug.print("config: {d}-bit groups of {d}, tied head {}, mrope {any} interleaved {}, max position {d}, end ids {any}\n", .{ s.bits, s.group, ck.config.tied, s.mrope.section, s.mrope.interleaved, s.max_position, ck.config.eos });
    const kinds = try gpa.alloc(u8, s.n_layers);
    defer gpa.free(kinds);
    for (0..s.n_layers) |i| kinds[i] = if (s.full(i)) 'F' else 'L';
    std.debug.print("layers: {s} (F full attention, L gated delta-net)\n", .{kinds});
    std.debug.print("files: {d} model, {d} mtp side; tensors in the model table: {d}\n", .{ ck.files.model.len, ck.files.side.len, ck.main.where.count() });
    var scratch: std.heap.ArenaAllocator = .init(gpa);
    defer scratch.deinit();
    const embed = try ck.embed(scratch.allocator());
    _ = try ck.finalNorm(scratch.allocator());
    const head = try ck.head(scratch.allocator());
    std.debug.print("embedding: {d} rows, {d} MiB; head: {s}\n", .{ embed.rows(), embed.bytes() >> 20, if (head == null) "the embedding" else "its own" });
    var full: usize = 0;
    var routed: usize = 0;
    for (0..s.n_layers) |i| {
        var layer = try ck.layer(i);
        defer layer.deinit();
        const mlp = switch (layer.body) {
            .full => |f| f.mlp,
            .linear => |l| l.mlp,
        };
        full += @intFromBool(layer.body == .full);
        routed += @intFromBool(mlp == .routed);
        if (i == 0) switch (mlp) {
            .routed => |r| std.debug.print("layer 0 mlp: routed, {d} experts + shared slot, width {d}, top-{d}\n", .{ r.experts.count - 1, r.experts.width, r.top_k }),
            .dense => std.debug.print("layer 0 mlp: dense\n", .{}),
        };
    }
    std.debug.print("every layer resolved: {d} full attention, {d} gated delta-net, {d} routed MLPs\n", .{ full, s.n_layers - full, routed });
    if (try ck.mtp()) |m| {
        var mtp = m;
        defer mtp.deinit();
        std.debug.print("mtp head: present, own head {}, mlp {s}, gated {}\n", .{ mtp.head != null, if (mtp.mlp) |x| @tagName(x) else "none", mtp.gated });
    } else std.debug.print("mtp head: absent\n", .{});
    return 0;
}
