//! A draft head's input rows from the target: the prompt's final normed rows, then each verify's kept rows.

const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const be = lanes.backend;

/// The tap id a head asks for to read the target's final normed rows (no layer's residual).
pub const final_tap: u32 = 0xffff_ffff;

/// The prompts' normed rows kept for drafting streams, until the stream's first verify.
pub const Held = struct {
    map: std.AutoHashMapUnmanaged(*lanes.Stream, cuda.DeviceBuffer) = .empty,
    on: bool = false, // a head asked for the final rows

    pub fn deinit(h: *Held, gpa: std.mem.Allocator) void {
        var it = h.map.valueIterator();
        while (it.next()) |b| b.free();
        h.map.deinit(gpa);
    }

    /// The stream's room for its prompt's rows, replacing any it held.
    pub fn make(h: *Held, gpa: std.mem.Allocator, d: *const cuda.Driver, s: *lanes.Stream, bytes: usize) !u64 {
        h.drop(s);
        var b = try cuda.DeviceBuffer.alloc(d, bytes);
        errdefer b.free();
        try h.map.put(gpa, s, b);
        return b.ptr;
    }

    pub fn drop(h: *Held, s: *lanes.Stream) void {
        if (h.map.fetchRemove(s)) |kv| {
            var b = kv.value;
            b.free();
        }
    }

    /// The taps a head may ask for: the final rows alone.
    pub fn prepare(h: *Held, taps: []const u32) !void {
        if (taps.len != 1 or taps[0] != final_tap) return error.UnsupportedDraftTaps;
        h.on = true;
    }

    /// Rows [start, start + count) of the stream's prompt, as long as they are held.
    pub fn prompt(h: *const Held, s: *lanes.Stream, start: u64, count: u32, row: u32) !?be.Features {
        const b = h.map.get(s) orelse return null;
        if ((start + count) * row > b.len) return error.FeatureRowsPastPrompt;
        return .{ .buffer = b.ptr, .offset = start * row, .rows = count, .row_bytes = row, .space = .device, .dtype = .bf16 };
    }
};
