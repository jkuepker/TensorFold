//! tf-qwen35-upload MODEL [--device N] [--digests] [--mtp]: a Qwen3.5-family MoE checkpoint onto the GPU, its bytes resident.
//! --digests prints "name sha256 bytes" for every uploaded tensor, the lines the oracle's weights.txt holds;
//! --mtp also uploads the MTP head and prints (with --digests) the lines of the oracle's mtp_weights.txt.

const std = @import("std");
const cuda = @import("cuda");
const qwen = @import("qwen35_cuda");

const usage = "usage: tf-qwen35-upload MODEL [--device N] [--digests] [--mtp]\n";
const Sha256 = std.crypto.hash.sha2.Sha256;

/// The device bytes of every tensor, hashed in order.
fn digests(gpa: std.mem.Allocator, d: *const cuda.Driver, named: []const qwen.weights.Named, s: cuda.Stream) !void {
    try s.synchronize();
    const chunk = 64 << 20;
    const host = try gpa.alloc(u8, chunk);
    defer gpa.free(host);
    var buf: [1 << 12]u8 = undefined;
    var out: std.Io.Writer = .fixed(&buf);
    for (named) |n| {
        var h = Sha256.init(.{});
        const dev: cuda.DeviceBuffer = .{ .d = d, .ptr = n.ptr, .len = n.len, .borrowed = true };
        var at: usize = 0;
        while (at < n.len) : (at += chunk) {
            const part = host[0..@min(chunk, n.len - at)];
            try dev.download(at, part);
            h.update(part);
        }
        const hex = std.fmt.bytesToHex(h.finalResult(), .lower);
        out.end = 0;
        try out.print("{s} {s} {d}\n", .{ n.name, hex, n.len });
        std.debug.print("{s}", .{out.buffered()});
    }
}

fn category(name: []const u8) usize {
    if (std.mem.endsWith(u8, name, ".moe.router")) return 1;
    if (std.mem.endsWith(u8, name, ".moe.up") or std.mem.endsWith(u8, name, ".moe.down")) return 2;
    if (std.mem.startsWith(u8, name, "embed.") or std.mem.startsWith(u8, name, "head.")) return 3;
    return 0;
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var device: c_int = 0;
    var want_digests = false;
    var want_mtp = false;
    var i: usize = 2;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--digests")) {
            want_digests = true;
        } else if (std.mem.eql(u8, args[i], "--mtp")) {
            want_mtp = true;
        } else if (std.mem.eql(u8, args[i], "--device") and i + 1 < args.len) {
            i += 1;
            device = try std.fmt.parseInt(c_int, args[i], 10);
        } else {
            std.debug.print("{s}", .{usage});
            return 2;
        }
    }
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, device);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(ctx.d, true);
    defer stream.deinit();
    const discrete = try ctx.attribute(.integrated) == 0;
    var kernels = try qwen.pack.Kernels.load(ctx.d, stream, discrete);
    defer kernels.deinit();
    const clock = std.Io.Clock.awake;
    const t0 = clock.now(init.io).toNanoseconds();
    var w = try qwen.weights.load(gpa, init.io, &kernels, args[1]);
    defer w.deinit();
    try stream.synchronize();
    const seconds = @as(f64, @floatFromInt(clock.now(init.io).toNanoseconds() - t0)) / 1e9;
    var sums = [_]usize{ 0, 0, 0, 0 };
    for (w.named.items) |n| sums[category(n.name)] += n.len;
    const gib = 1 << 30;
    std.debug.print("resident: {d} bytes ({d:.2} GiB) in {d:.1}s over {d} layers\n", .{ w.bytes(), @as(f64, @floatFromInt(w.bytes())) / gib, seconds, w.layers.len });
    std.debug.print("resident: trunk {d}, routers {d}, expert stacks {d}, embedding and head {d}\n", .{ sums[0], sums[1], sums[2], sums[3] });
    if (want_digests) try digests(gpa, ctx.d, w.named.items, stream);
    if (want_mtp) {
        var head = try qwen.mtp_weights.load(gpa, init.io, &kernels, args[1], w.base.g);
        defer head.deinit();
        try stream.synchronize();
        std.debug.print("mtp head: {d} bytes, {d} draft ids\n", .{ head.bytes(), head.ids.len });
        if (want_digests) {
            try digests(gpa, ctx.d, head.named.items, stream);
            var h = Sha256.init(.{});
            for (head.ids) |id| h.update(&std.mem.toBytes(@as(i64, id)));
            const hex = std.fmt.bytesToHex(h.finalResult(), .lower);
            std.debug.print("mtp.ids {s} {d}\n", .{ hex, head.ids.len * 8 });
        }
    }
    return 0;
}
