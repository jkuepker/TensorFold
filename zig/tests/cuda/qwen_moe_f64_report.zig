//! Per-output, per-phase float64 error lines and gates for the Qwen3.6 MoE tests (router, experts, projections).
const std = @import("std");
const check = @import("check.zig");
const q = @import("qwen_f64.zig");

/// One kernel run's elements against their float64 reference and bound, `cols` to a row.
pub const Run = struct { got: []const f64, want: []const f64, tol: []const f64 };

/// Feeds a pair: native into its statistics, and with a Triton run (own reference: its picks may differ) Triton's.
pub fn feed(p: *q.Pair, nat: Run, tri: ?Run) !void {
    try p.n.addSlices(nat.got, nat.want, nat.tol);
    const t = tri orelse return;
    try p.np.addSlices(nat.got, nat.want, nat.tol);
    try p.t.addSlices(t.got, t.want, t.tol);
}

fn gates(p: *const q.Pair) !void {
    const n = &p.n;
    if (n.fails > 0) {
        const f = n.first.?;
        try check.expect(false, "{s}: {d} elements past their bound, first in column {d}: got {e:.6}, want {e:.6}, bound {e:.3}", .{ n.name, n.fails, f.col, f.got, f.want, f.bound });
    }
    if (p.t.fails > 0) {
        const f = p.t.first.?;
        try check.expect(false, "{s}: {d} triton elements past their bound, first in column {d}: got {e:.6}, want {e:.6}, bound {e:.3}", .{ n.name, p.t.fails, f.col, f.got, f.want, f.bound });
    }
    if (p.worseCol()) |c| {
        try check.expect(false, "{s}: column {d} native abs {e:.3} rel {e:.3} worse than Triton's {e:.3} {e:.3}", .{ n.name, c, p.np.abs[c], p.np.rel[c], p.t.abs[c], p.t.rel[c] });
    }
}

fn pairLine(prefix: []const u8, p: *const q.Pair) void {
    std.debug.print("{s} {s} cols={d}", .{ prefix, p.n.name, p.n.abs.len });
    p.n.print("native");
    if (p.t.count > 0) p.t.print("triton") else std.debug.print(" triton none", .{});
}

/// One RESULT line per phase the case ran in (native, then Triton), then the gates of each: bound, native no worse.
pub fn show(c: *const q.Phased) !void {
    var failed = false;
    for (&c.by) |*p| {
        if (p.n.count == 0) continue;
        pairLine("RESULT", p);
        std.debug.print("\n", .{});
        try q.gate(&failed, gates(p));
    }
    if (failed) return error.TestFailed;
}

/// One output over a whole run: both phases, every case folded in.
pub const Out = struct {
    bound: []const u8,
    run: q.Phased,

    pub fn init(a: std.mem.Allocator, name: []const u8, bound: []const u8, cols: usize) !Out {
        return .{ .bound = bound, .run = try q.Phased.init(a, name, cols) };
    }

    pub fn deinit(o: *Out) void {
        o.run.deinit();
    }

    pub fn fold(o: *Out, c: *const q.Phased) void {
        for (&o.run.by, &c.by) |*to, *from| {
            to.n.merge(&from.n);
            to.np.merge(&from.np);
            to.t.merge(&from.t);
        }
    }

    /// The summary lines: worst columns and aggregates per phase, a phase nothing ran in said so.
    pub fn table(o: *const Out) void {
        for (&o.run.by) |*p| {
            if (p.n.count == 0) {
                std.debug.print("TABLE {s}: no rows in this phase\n", .{p.n.name});
                continue;
            }
            pairLine("TABLE", p);
            std.debug.print("; bound {s}\n", .{o.bound});
        }
    }
};

/// Negative control: the element of the largest reference, moved 4 bounds away, must be refused.
pub fn control(a: std.mem.Allocator, name: []const u8, got: []const f64, want: []const f64, tol: []const f64, cols: usize) !void {
    var at: usize = 0;
    for (want, 0..) |w, i| {
        if (@abs(w) > @abs(want[at])) at = i;
    }
    const bad = try a.dupe(f64, got);
    defer a.free(bad);
    bad[at] += 4 * tol[at];
    var s = try q.Stat.init(a, name, cols, false);
    defer s.deinit();
    try s.addSlices(bad, want, tol);
    std.debug.print("CONTROL {s}: element {d} moved 4 bounds -> {d:.3} of bound, {s}\n", .{ name, at, q.Stat.worst(s.ratio).val, if (s.fails > 0) "refused" else "ACCEPTED" });
    try check.expect(s.fails > 0, "{s}: a perturbed output is refused", .{name});
}
