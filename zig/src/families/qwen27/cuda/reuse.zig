//! A stream's kept prompt state for the prompt cache: recurrences, conv windows and the attention rows before `at`.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const c = @import("shape.zig");
const st = @import("state.zig");
const kern = @import("kernels.zig");
const engine = @import("engine.zig");
const features = @import("features.zig");
const Engine = engine.Engine;

/// cuMemAlloc hands out whole 2 MiB pages: a kept state is charged for them.
const page: usize = 2 << 20;

/// The device bytes a kept state of `at` tokens takes: every DeltaNet layer's recurrence and conv window, every attention layer's keys and values, and with `rows` the prompt's final normed rows a draft head reads.
pub fn bytes(g: c.Geometry, at: usize, rows: bool) usize {
    return std.mem.alignForward(usize, stateLen(g, at) + if (rows) at * g.hidden * 2 else 0, page);
}

/// One kept state: the stream's layers laid end to end in a device buffer, then (when a drafting stream kept it) the prompt's normed rows.
pub const Saved = struct { buf: cuda.DeviceBuffer, at: usize, gpa: std.mem.Allocator, rows: ?u64 = null };

const Part = struct { dev: u64, len: usize };

/// The bytes the layers' parts take, before any rows.
fn stateLen(g: c.Geometry, at: usize) usize {
    var n: usize = 0;
    for (0..g.layers) |i| n += if (c.linear(i)) g.deltaBytes() + g.convBytes() else 2 * at * g.kvInner() * 2;
    return n;
}

/// Copies each layer's part between the stream (`into` false: out of it) and `base`, the first `at` attention rows.
fn move(ops: kern.Ops, g: c.Geometry, seq: *const st.Seq, base: u64, at: usize, into: bool) !void {
    var off: u64 = 0;
    const row = at * g.kvInner() * 2;
    for (0..g.layers) |i| {
        const parts = if (c.linear(i)) [2]Part{ .{ .dev = seq.rec[i], .len = g.deltaBytes() }, .{ .dev = seq.conv[i], .len = g.convBytes() } } else [2]Part{ .{ .dev = seq.keys[i], .len = row }, .{ .dev = seq.values[i], .len = row } };
        for (parts) |p| {
            if (into) try ops.copy(p.dev, base + off, p.len) else try ops.copy(base + off, p.dev, p.len);
            off += p.len;
        }
    }
}

/// The stream's state after `at` prompt tokens, copied on the engine's stream (the pass stands at `at`).
/// `rows` is the stream's held normed rows (at least `at` of them), kept beside the state.
pub fn save(gpa: std.mem.Allocator, ops: kern.Ops, g: c.Geometry, seq: *const st.Seq, at: usize, rows: ?u64) !*Saved {
    if (seq.pos != at) return error.NotAtMark;
    const saved = try gpa.create(Saved);
    errdefer gpa.destroy(saved);
    saved.* = .{ .buf = cuda.DeviceBuffer.alloc(ops.k.d, bytes(g, at, rows != null)) catch |err| return if (err == error.OutOfMemory) err else error.OutOfDeviceMemory, .at = at, .gpa = gpa };
    errdefer saved.buf.free();
    try move(ops, g, seq, saved.buf.ptr, at, false);
    if (rows) |r| {
        saved.rows = saved.buf.ptr + stateLen(g, at);
        try ops.copy(saved.rows.?, r, at * g.hidden * 2);
    }
    try ops.s.synchronize();
    return saved;
}

/// The stream's state made `saved`'s: its prompt pass goes on from `saved.at`.
/// `rows`, a drafting stream's held rows, get the kept ones back (the state must have kept them).
pub fn restore(ops: kern.Ops, g: c.Geometry, seq: *st.Seq, saved: *const Saved, rows: ?u64) !void {
    if (saved.at >= seq.capacity) return error.PromptTooLong;
    try move(ops, g, seq, saved.buf.ptr, saved.at, true);
    if (rows) |r| try ops.copy(r, saved.rows orelse return error.NoKeptRows, saved.at * g.hidden * 2);
    try ops.s.synchronize();
    seq.pos = saved.at;
}

pub fn drop(saved: *Saved) void {
    saved.buf.free();
    saved.gpa.destroy(saved);
}

/// The streams' states by stream: what the cache's copies read and write.
pub const Seqs = std.AutoHashMapUnmanaged(*lanes.Stream, *st.Seq);

/// What the prompt cache copies: the engine, and the lane backend's streams (a state belongs to the stream whose pass keeps it).
pub const Target = struct { e: *Engine, seqs: *const Seqs, held: *const features.Held };

fn target(p: *anyopaque) *const Target {
    return @ptrCast(@alignCast(p));
}

fn ownerSeq(t: *const Target, owner: ?*anyopaque) !*st.Seq {
    const s: *lanes.Stream = @ptrCast(@alignCast(owner orelse return error.NoOwner));
    return t.seqs.get(s) orelse error.UnknownQwenStream;
}

pub fn snapBytes(ptr: *anyopaque, at: u32) u64 {
    const t = target(ptr);
    return bytes(t.e.w.g, at, t.held.on);
}

pub fn snapSave(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    const t = target(ptr);
    const rows = if (t.held.map.get(@ptrCast(@alignCast(owner orelse return error.NoOwner)))) |b| b.ptr else null;
    return try save(t.e.gpa, t.e.ops(), t.e.w.g, try ownerSeq(t, owner), at, rows);
}

pub fn snapRestore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    const t = target(ptr);
    try restore(t.e.ops(), t.e.w.g, try ownerSeq(t, owner), @ptrCast(@alignCast(saved)), null);
}

pub fn snapDrop(_: *anyopaque, saved: *anyopaque) void {
    drop(@ptrCast(@alignCast(saved)));
}

/// The stream's prompt pass: its kept state restored when the cache found one, then the prompt's rows, the cache's marks kept on the way.
/// `rows` (a drafting stream's held rows) take the kept state's rows too; a state kept without them is left alone.
pub fn prefill(e: *Engine, s: *lanes.Stream, seq: *st.Seq, rows: ?u64) !void {
    const ids = s.prompt();
    var from: usize = 0;
    s.cached = 0;
    if (s.reuse.saved) |p| {
        const saved: *const Saved = @ptrCast(@alignCast(p));
        if (rows != null and saved.rows == null) {
            // a state kept by a stream without drafts has no rows for the head: the pass runs whole, the entry stays
        } else if (saved.at < ids.len) {
            if (restore(e.ops(), e.w.g, seq, saved, rows)) |_| {
                from = saved.at;
                s.cached = @intCast(saved.at);
            } else |_| s.reuse_failed = true;
        } else s.reuse_failed = true;
    }
    const keep: ?engine.Keep = if (s.reuse.marks.len == 0) null else .{ .at = s.reuse.marks, .ctx = s, .call = keptAt };
    try e.prefillFrom(seq, ids, from, keep, rows);
}

/// The pass stands at a mark: the lane host copies the stream's state there.
fn keptAt(ctx: *anyopaque, at: u32) void {
    const s: *lanes.Stream = @ptrCast(@alignCast(ctx));
    if (s.reuse.hook) |k| k.at(k.ptr, s, at);
}

test "a kept state is its DeltaNet layers' fixed part, the attention rows and a head's rows, in whole pages" {
    const g = c.the_35b_a3b;
    const fixed = 30 * (g.deltaBytes() + g.convBytes());
    try std.testing.expectEqual(std.mem.alignForward(usize, fixed, page), bytes(g, 0, false));
    const per = 10 * 2 * g.kvInner() * 2;
    try std.testing.expectEqual(std.mem.alignForward(usize, fixed + 4096 * per, page), bytes(g, 4096, false));
    try std.testing.expect(bytes(g, 8192, false) > bytes(g, 4096, false));
    try std.testing.expectEqual(std.mem.alignForward(usize, fixed + 4096 * (per + g.hidden * 2), page), bytes(g, 4096, true)); // a drafting stream's rows ride along
}
