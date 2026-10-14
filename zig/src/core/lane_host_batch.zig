//! A lane host's admission round: every queued request the lanes have room for, the rest prefilled together.
const std = @import("std");
const lanes = @import("lanes");
const LaneHost = @import("lane_host.zig").LaneHost;
const lone = @import("lane_host_lone.zig");
const Job = LaneHost.Job;

/// Every queued request the lanes have room for: a lone one to the backend's driver, the rest prefilled together.
pub fn all(h: *LaneHost) void {
    var batch: std.ArrayList(*Job) = .empty;
    defer batch.deinit(h.gpa);
    while (true) {
        const job = switch (h.prepare()) {
            .none => break,
            .dropped => continue,
            .job => |j| j,
        };
        if (job.request.max_tokens == 0 and h.info_.warm_turns) {
            _ = h.warmPass(job, job.began);
            continue;
        }
        if (batch.items.len == 0 and lone.fits(h, job)) {
            _ = lone.run(h, job, job.began, false);
            continue;
        }
        if (h.core.fills()) {
            h.filling.append(h.gpa, job) catch {
                _ = h.drop(job, "out of memory");
            };
            continue;
        }
        batch.append(h.gpa, job) catch {
            _ = h.drop(job, "out of memory");
            continue;
        };
    }
    if (batch.items.len == 0) return;
    const streams = h.gpa.alloc(*lanes.Stream, batch.items.len) catch return dropAll(h, batch.items, "out of memory");
    defer h.gpa.free(streams);
    const errs = h.gpa.alloc(?anyerror, batch.items.len) catch return dropAll(h, batch.items, "out of memory");
    defer h.gpa.free(errs);
    for (streams, batch.items) |*s, job| s.* = &job.stream;
    h.core.addStreams(streams, errs) catch |e| return dropAll(h, batch.items, h.words(e));
    for (batch.items, errs) |job, err| {
        if (err) |e| {
            _ = switch (e) {
                error.Cancelled => h.cancel(job),
                error.LaneMemory => h.requeue(job),
                else => h.drop(job, h.words(e)),
            };
            continue;
        }
        h.prefilled(job, job.began);
        if (h.deliver(job)) h.remove(job);
    }
}

fn dropAll(h: *LaneHost, jobs: []const *Job, message: []const u8) void {
    for (jobs) |job| _ = h.drop(job, message);
}
