//! A stream's way into the rounds: its prompt pass whole or a chunk a round, then its first token and drafts.
const std = @import("std");
const Engine = @import("engine.zig").Engine;
const sm = @import("stream.zig");
const be = @import("backend.zig");
const ev = @import("events.zig");
const win = @import("windows.zig");
const trail = @import("trail.zig");
const Stream = sm.Stream;
const Feed = be.Feed;
const LogRow = @import("logprob.zig").Row;
const f = ev.f;
const str = trail.str;
const int = trail.int;

/// Prefill a stream, draw its first token and ask for its first drafts; it takes part from the next round.
pub fn add(e: *Engine, s: *Stream) !void {
    var err: [1]?anyerror = undefined;
    try addMany(e, &.{s}, &err);
    if (err[0]) |x| return x;
}

/// `add` for several streams: prompt passes shared when the backend has them, every first draft in one call.
pub fn addMany(e: *Engine, list: []const *Stream, errs: []?anyerror) !void {
    _ = e.arena.reset(.retain_capacity);
    @memset(errs, null);
    for (list) |s| try trail.event(e, &.{ f("ev", str("add")), f("stream", str(s.id)) });
    if (list.len > 1 and e.backend.vtable.prefill_many != null) {
        try e.backend.vtable.prefill_many.?(e.backend.ptr, list, errs);
    } else for (list, errs) |s, *err| e.backend.prefill(s) catch |x| {
        err.* = x;
    };
    try openedMany(e, list, errs);
}

/// After the prompt pass: the first token drawn and committed, the first drafts asked; rounds from the next.
pub fn opened(e: *Engine, s: *Stream) !void {
    var err: [1]?anyerror = .{null};
    try openedMany(e, &.{s}, &err);
    if (err[0]) |x| return x;
}

/// `opened` for several streams, their first drafts asked in one call; a failed one has its error in `errs`.
pub fn openedMany(e: *Engine, list: []const *Stream, errs: []?anyerror) !void {
    const a = e.arena.allocator();
    const feeds = try a.alloc(Feed, list.len);
    const drawn = try a.alloc(u64, list.len);
    const asked = try a.alloc(?u32, list.len);
    var requests: std.ArrayList(be.DraftRequest) = .empty;
    for (list, errs, feeds, drawn, asked) |s, *err, *feed, *handle, *ask| {
        ask.* = null;
        if (err.*) |x| {
            if (x == error.Cancelled) e.backend.release(s); // the host finishes a cancelled stream without the core
            continue;
        }
        if (s.isCancelled()) { // cancelled in its last chunk: no first token
            e.backend.release(s);
            err.* = error.Cancelled;
            continue;
        }
        s.context.shrinkRetainingCapacity(s.prompt_len);
        s.rows.clearRetainingCapacity();
        s.pending = null;
        s.cache_len = s.prompt_len;
        const position: u64 = s.prompt_len;
        handle.* = e.drawFirst(s, position) catch |x| {
            err.* = x; // the host discards it, as after a failed prompt pass
            continue;
        };
        feed.* = .{ .handle = handle.* };
        if (try e.forcedNext(s)) |t| feed.* = .{ .value = t };
        if (e.cfg.family_mtp and s.drafts) {
            // the head reads the prompt's last row and the first token, and drafts the one after it
            const d: u32 = @intCast(try e.rule.depth(win.who(s)));
            ask.* = d;
            try requests.append(a, .{ .stream = s, .follow = &.{}, .first = feed.*, .rows = null, .start = s.prompt_len, .position = position + 1, .depth = d });
        } else if (e.cfg.pipelined and s.logprobs == null and s.grammar == null) {
            try e.queueNext(s, feed.*);
        }
    }
    if (requests.items.len > 0) try e.backend.draft(requests.items);
    for (list, errs, feeds, drawn, asked) |s, err, feed, handle, ask| {
        if (err != null) continue;
        const position: u64 = s.prompt_len;
        if (ask) |d| {
            s.dropHeld(e.gpa);
            s.next = .{ .count = d };
            if (e.backend.vtable.tree) |tree| if (try tree(e.backend.ptr, s, e.gpa)) |held| {
                s.next = held; // a tree head's first drafts as host tokens, as every later round's
            };
        }
        const value = try e.readFeed(feed);
        if (ask) |d| try trail.event(e, &.{ f("ev", str("draft")), f("stream", str(s.id)), f("depth", int(d)), f("position", int(position + 1)), f("follow", .{ .u32s = &.{value} }), f("rows", .null) });
        if (e.log != null) {
            const first = if (feed == .handle) value else try e.backend.read(handle);
            try trail.event(e, &.{ f("ev", str("first")), f("stream", str(s.id)), f("position", int(position)), f("drawn", int(first)), f("token", int(value)) });
        }
        var first_row: [1]LogRow = undefined;
        if (s.logprobs != null) first_row[0] = (try e.backend.firstRow(s)).forToken(value);
        _ = try s.commit(e.gpa, &.{value}, if (s.logprobs != null) &first_row else &.{});
        s.pending = value;
        try trail.resolve(e);
        if (s.finished) {
            try trail.finish(e, s);
            e.release(s);
            continue;
        }
        try e.live.append(e.gpa, s);
    }
}

/// Whether the backend can run a prompt pass a chunk at a time between rounds (`fillStream`).
pub fn fills(e: *const Engine) bool {
    return e.backend.vtable.prefill_step != null;
}

/// One chunk of a stream's prompt pass; true once the stream is opened and joins the next round.
pub fn fillStream(e: *Engine, s: *Stream, first: bool) !bool {
    _ = e.arena.reset(.retain_capacity);
    if (first) try trail.event(e, &.{ f("ev", str("add")), f("stream", str(s.id)) });
    const chunk = e.backend.vtable.prefill_step orelse return error.NoPrefillSteps;
    const done = chunk(e.backend.ptr, s) catch |err| {
        if (err == error.Cancelled) e.backend.release(s); // the host finishes a cancelled stream without the core
        return err;
    };
    if (!done) return false;
    try opened(e, s);
    return true;
}
