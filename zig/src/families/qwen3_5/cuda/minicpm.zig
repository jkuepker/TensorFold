//! MiniCPM-V 4.7 (minicpmv4_7), text only, on CUDA: a Qwen3.6-35B-A3B fine-tune served by Qwen3.6's engine.

const std = @import("std");
const cuda = @import("cuda");
const base = @import("native.zig");

pub const model_type = "minicpmv4_7";
pub const formats = base.formats;
pub const default_context = base.default_context;
pub const max_segments = base.max_segments;
pub const prompt_rows = base.prompt_rows;
pub const max_lanes = base.max_lanes;
pub const Options = base.Options;
pub const Loaded = base.Loaded;
pub const snap_bytes = base.snap_bytes;
pub const snap_save = base.snap_save;
pub const snap_restore = base.snap_restore;
pub const snap_drop = base.snap_drop;
pub const snap_rules = base.snap_rules;
pub const refusal = base.refusal;

/// The MTP layer's file name in Qwen3.6's 4-bit repo, and the repos that hold it (old and new org names).
pub const mtp_file = "mtp-4bit.safetensors";
const repos = [_][]const u8{ "TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP", "Vontra/Qwen3.6-35B-A3B-MLX-4bit-MTP" };

/// Where the draft head comes from: a file, or nothing to draft with.
pub const Found = union(enum) { path: []const u8, off, none };

/// TF_QWEN36_MTP (a path, or 0 for none), the checkpoint's own mtp file, else the cached Qwen3.6 repo's file.
pub fn pick(a: std.mem.Allocator, io: std.Io, dir: []const u8, named: ?[]const u8, hub: ?[]const u8) !Found {
    if (named) |n| if (n.len > 0) {
        if (std.mem.eql(u8, n, "0")) return .off;
        if (!isFile(io, n)) return error.MtpFileMissing;
        return .{ .path = n };
    };
    const own = try std.fs.path.join(a, &.{ dir, mtp_file });
    if (isFile(io, own)) return .{ .path = own };
    const root = hub orelse return .none;
    for (repos) |repo| if (try cachedFile(a, io, root, repo)) |p| return .{ .path = p };
    return .none;
}

/// The newest snapshot of `repo` in the hub cache that holds the MTP file.
fn cachedFile(a: std.mem.Allocator, io: std.Io, root: []const u8, repo: []const u8) !?[]const u8 {
    const folder = try std.mem.replaceOwned(u8, a, repo, "/", "--");
    const snapshots = try std.fs.path.join(a, &.{ root, try std.fmt.allocPrint(a, "models--{s}", .{folder}), "snapshots" });
    var d = std.Io.Dir.cwd().openDir(io, snapshots, .{ .iterate = true }) catch return null;
    defer d.close(io);
    var best: ?[]const u8 = null;
    var newest: i96 = std.math.minInt(i96);
    var it = d.iterate();
    while (try it.next(io)) |e| {
        if (e.kind != .directory and e.kind != .sym_link) continue;
        const file = try std.fs.path.join(a, &.{ snapshots, e.name, mtp_file });
        const st = std.Io.Dir.cwd().statFile(io, file, .{}) catch continue;
        if (best == null or st.mtime.nanoseconds > newest) {
            best = file;
            newest = st.mtime.nanoseconds;
        }
    }
    return best;
}

fn isFile(io: std.Io, path: []const u8) bool {
    const st = std.Io.Dir.cwd().statFile(io, path, .{}) catch return false;
    return st.kind == .file;
}

fn getenv(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}

/// The Hugging Face hub cache: HF_HUB_CACHE, else HF_HOME/hub, else ~/.cache/huggingface/hub.
fn hubCache(a: std.mem.Allocator) !?[]const u8 {
    if (getenv("HF_HUB_CACHE")) |c| if (c.len > 0) return c;
    if (getenv("HF_HOME")) |h| if (h.len > 0) return try std.fs.path.join(a, &.{ h, "hub" });
    const home = getenv("HOME") orelse return null;
    return try std.fs.path.join(a, &.{ home, ".cache", "huggingface", "hub" });
}

/// The engine for a MiniCPM-V 4.7 checkpoint: Qwen3.6's, its drafts from Qwen3.6's MTP layer.
pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    var side: ?[]const u8 = null;
    if (o.drafts) switch (pick(arena.allocator(), io, dir, getenv("TF_QWEN36_MTP"), try hubCache(arena.allocator())) catch |e| {
        if (e == error.MtpFileMissing) std.log.err("TF_QWEN36_MTP names a file that is not there", .{});
        return e;
    }) {
        .path => |p| side = p,
        .off, .none => {
            std.log.err("MiniCPM-V 4.7 ships no MTP layer: drafts read Qwen3.6's ({s}, 0.5 GB). Fetch it with `hf download TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP {s}`, or point TF_QWEN36_MTP at it, or serve with --no-drafts", .{ mtp_file, mtp_file });
            return error.MiniCpmNeedsQwen36Mtp;
        },
    };
    return base.openWith(gpa, io, ctx, dir, kernels, o, side);
}

/// A request this engine refuses, in words; null: none of its own.
pub fn explain(p: ?*anyopaque, err: anyerror) ?[]const u8 {
    return base.explain(p, err);
}

test "TF_QWEN36_MTP wins, 0 turns drafting off, a missing named file is refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    const root = try std.fmt.allocPrint(a, ".tf-minicpm-test-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds()});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    try std.Io.Dir.cwd().createDirPath(io, root);
    const named = try std.fs.path.join(a, &.{ root, "named.safetensors" });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = named, .data = "x" });
    try std.testing.expectEqualStrings(named, (try pick(a, io, root, named, null)).path);
    try std.testing.expect((try pick(a, io, root, "0", null)) == .off);
    try std.testing.expectError(error.MtpFileMissing, pick(a, io, root, "/nonexistent/mtp.safetensors", null));
    try std.testing.expect((try pick(a, io, root, "", null)) == .none);
}

test "the checkpoint's own file comes before the hub cache's" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    const root = try std.fmt.allocPrint(a, ".tf-minicpm-test-{d}", .{std.Io.Clock.awake.now(io).toNanoseconds() + 1});
    defer std.Io.Dir.cwd().deleteTree(io, root) catch {};
    const hub = try std.fs.path.join(a, &.{ root, "hub" });
    const snapshot = try std.fs.path.join(a, &.{ hub, "models--Vontra--Qwen3.6-35B-A3B-MLX-4bit-MTP", "snapshots", "abc" });
    try std.Io.Dir.cwd().createDirPath(io, snapshot);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ snapshot, mtp_file }), .data = "x" });
    const cached = (try pick(a, io, root, null, hub)).path;
    try std.testing.expect(std.mem.endsWith(u8, cached, "snapshots/abc/" ++ mtp_file));
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fs.path.join(a, &.{ root, mtp_file }), .data = "y" });
    try std.testing.expectEqualStrings(try std.fs.path.join(a, &.{ root, mtp_file }), (try pick(a, io, root, null, hub)).path);
}
