//! Streams admitted in one round open together only for a backend that keeps each first row apart (lane_host_batch).
const std = @import("std");
const lanes = @import("lanes");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Reason = api.Reason;
const Request = api.Request;

const Box = struct {
    mutex: std.Io.Mutex = .init,
    done: ?Reason = null,
    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *@This() = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(std.testing.io);
        defer b.mutex.unlock(std.testing.io);
        if (e.* == .finished) b.done = e.finished.reason;
    }
    fn wait(b: *@This()) Reason {
        while (true) {
            b.mutex.lockUncancelable(std.testing.io);
            const d = b.done;
            b.mutex.unlock(std.testing.io);
            if (d) |r| return r;
            std.Io.sleep(std.testing.io, .fromMilliseconds(1), .awake) catch {};
        }
    }
};

/// How many first tokens the backend had drawn when each prompt pass began.
const Passes = struct {
    target: *lanes.fake.Fake,
    drawn: [2]usize = .{ 99, 99 },
    count: usize = 0,
    fn hook(ctx: *anyopaque, _: *lanes.Stream, _: usize) void {
        const p: *@This() = @ptrCast(@alignCast(ctx));
        if (p.count < p.drawn.len) p.drawn[p.count] = p.target.drawn.items.len;
        p.count += 1;
    }
};

/// Two requests queued before the host starts, so one admission round takes both; what each prompt pass saw.
fn twoRequests(open_together: bool) ![2]usize {
    const gpa = std.testing.allocator;
    var cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true, .open_together = open_together }, 8, 7);
    defer cfg.deinit(gpa);
    var target: lanes.fake.Fake = .{ .gpa = gpa };
    defer target.deinit();
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(gpa, &cfg, target.backend(), clock.clock());
    defer core.deinit();
    var passes: Passes = .{ .target = &target };
    target.prefill_hook = Passes.hook;
    target.prefill_hook_ctx = &passes;
    var host = LaneHost.init(gpa, std.testing.io, &core, .{ .lanes = 2 });
    const prompt = [_]u32{ 3, 1, 4, 1, 5, 9, 2, 6 };
    const request: Request = .{ .prompt = &prompt, .max_tokens = 4 };
    var a: Box = .{};
    var b: Box = .{};
    const e = host.engine();
    try e.submit(1, &request, .{ .ctx = &a, .event = Box.event });
    try e.submit(2, &request, .{ .ctx = &b, .event = Box.event });
    try host.start();
    defer host.stop();
    try std.testing.expectEqual(Reason.length, a.wait());
    try std.testing.expectEqual(Reason.length, b.wait());
    try std.testing.expectEqual(@as(usize, 2), passes.count);
    return passes.drawn;
}

test "a backend that does not open together has each stream opened before the next one's prompt pass" {
    const drawn = try twoRequests(false);
    try std.testing.expectEqual(@as(usize, 0), drawn[0]);
    try std.testing.expect(drawn[1] > 0); // the first stream's first token was drawn before the second prompt pass
}

test "a backend that opens together has both prompt passes run before either stream's first token" {
    try std.testing.expectEqual([2]usize{ 0, 0 }, try twoRequests(true));
}
