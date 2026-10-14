//! Float64 reference scaffolding for the Qwen3.6 precision tests: bf16 decode, ulp spacing, per-column statistics.

const std = @import("std");
const check = @import("check.zig");

pub fn bf(x: u16) f64 {
    return @as(f32, @bitCast(@as(u32, x) << 16));
}

pub fn bf16Of(v: f32) u16 {
    const u: u32 = @bitCast(v);
    if (std.math.isNan(v)) return 0x7fc0;
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

pub fn bfAt(raw: []const u8, i: usize) f64 {
    return bf(std.mem.readInt(u16, raw[2 * i ..][0..2], .little));
}

pub fn f32At(raw: []const u8, i: usize) f64 {
    return @as(f32, @bitCast(std.mem.readInt(u32, raw[4 * i ..][0..4], .little)));
}

/// The spacing of bf16 values around `x` (8 significand bits), at least the smallest subnormal step.
pub fn ulpBf16(x: f64) f64 {
    const a = @abs(x);
    if (!(a > 0) or !std.math.isFinite(a)) return 0x1p-133;
    return @max(@exp2(@as(f64, @floatFromInt(std.math.frexp(a).exponent - 8))), 0x1p-133);
}

pub const flt_min: f64 = 0x1p-126;

/// The column whose elements the negative control pushes off the reference (TF_QWEN_F64_PERTURB=<column>).
pub fn controlColumn() ?usize {
    const v = std.c.getenv("TF_QWEN_F64_PERTURB") orelse return null;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch null;
}

pub const Stat = struct {
    gpa: std.mem.Allocator,
    name: []const u8,
    abs: []f64,
    rel: []f64,
    ratio: []f64,
    bound: []f64,
    inject: bool, // the negative control may move this statistic's elements (the native one only)
    control: ?usize,
    count: usize = 0,
    sq: f64 = 0, // sum of squared errors, for the aggregate RMS beside the worst column
    ratio_sum: f64 = 0,

    pub fn init(gpa: std.mem.Allocator, name: []const u8, cols: usize, inject: bool) !Stat {
        const cols_all = try gpa.alloc(f64, 4 * cols);
        @memset(cols_all, 0);
        return .{ .gpa = gpa, .name = name, .abs = cols_all[0..cols], .rel = cols_all[cols .. 2 * cols], .ratio = cols_all[2 * cols .. 3 * cols], .bound = cols_all[3 * cols ..], .inject = inject, .control = controlColumn() };
    }

    pub fn deinit(s: *Stat) void {
        const cols = s.abs.len;
        s.gpa.free(s.abs.ptr[0 .. 4 * cols]);
    }

    /// One element of column `col`: `got` against `want` within `bound`; relative error is over max(|want|, floor).
    pub fn add(s: *Stat, col: usize, got: f64, want: f64, bound: f64, floor: f64) void {
        var g = got;
        const b = @max(bound, 1e-300);
        if (s.inject and s.control != null and s.control.? == col) g += if (g >= want) 1.25 * b else -1.25 * b;
        const err = if (std.math.isFinite(g)) @abs(g - want) else std.math.inf(f64);
        s.count += 1;
        s.sq += err * err;
        s.ratio_sum += err / b;
        s.abs[col] = @max(s.abs[col], err);
        s.rel[col] = @max(s.rel[col], err / @max(@abs(want), floor));
        s.ratio[col] = @max(s.ratio[col], err / b);
        s.bound[col] = @max(s.bound[col], b);
    }
};

fn worst(v: []const f64) struct { at: usize, val: f64 } {
    var at: usize = 0;
    for (v, 0..) |x, i| if (x > v[at]) {
        at = i;
    };
    return .{ .at = at, .val = v[at] };
}

/// A native statistic over every element, the native one over the elements Triton also produced, and Triton's own.
pub const Pair = struct {
    n: Stat,
    np: Stat,
    t: Stat,

    pub fn init(gpa: std.mem.Allocator, name: []const u8, cols: usize) !Pair {
        return .{ .n = try Stat.init(gpa, name, cols, true), .np = try Stat.init(gpa, name, cols, true), .t = try Stat.init(gpa, name, cols, false) };
    }

    pub fn deinit(p: *Pair) void {
        p.n.deinit();
        p.np.deinit();
        p.t.deinit();
    }

    pub fn add(p: *Pair, col: usize, got: f64, tri_got: ?f64, want: f64, bound: f64, floor: f64) void {
        p.n.add(col, got, want, bound, floor);
        if (tri_got) |g| {
            p.np.add(col, got, want, bound, floor);
            p.t.add(col, g, want, bound, floor);
        }
    }

    /// One machine-readable line, then both gates: error within bound (when `strict`), native no worse than Triton.
    pub fn finish(p: *const Pair, bound_text: []const u8, strict: bool) !void {
        const n = &p.n;
        const a = worst(n.abs);
        const r = worst(n.rel);
        const q = worst(n.ratio);
        const bd = worst(n.bound);
        const cnt: f64 = @floatFromInt(@max(n.count, 1));
        std.debug.print("F64 {s} cols={d} n={d} native abs={e:.3} (col {d}) rel={e:.3} (col {d}) err/bound={d:.3} (col {d}) bound<={e:.3} rms={e:.3} mean_err/bound={d:.4}", .{ n.name, n.abs.len, n.count, a.val, a.at, r.val, r.at, q.val, q.at, bd.val, @sqrt(n.sq / cnt), n.ratio_sum / cnt });
        if (p.t.count > 0) {
            const t = &p.t;
            const ta = worst(t.abs);
            const tr = worst(t.rel);
            const tq = worst(t.ratio);
            const tc: f64 = @floatFromInt(t.count);
            std.debug.print(" triton abs={e:.3} (col {d}) rel={e:.3} (col {d}) err/bound={d:.3} rms={e:.3} mean_err/bound={d:.4} (n={d})", .{ ta.val, ta.at, tr.val, tr.at, tq.val, @sqrt(t.sq / tc), t.ratio_sum / tc, t.count });
        } else std.debug.print(" triton none", .{});
        std.debug.print(" bound=\"{s}\" strict={}\n", .{ bound_text, strict });
        var failed = false;
        if (strict and q.val > 1) {
            std.debug.print("FAIL {s}: column {d} is {d:.3} of its bound\n", .{ n.name, q.at, q.val });
            failed = true;
        }
        if (p.t.count > 0) for (p.np.abs, p.t.abs, p.np.rel, p.t.rel, 0..) |na, ta, nr, tr, c| {
            if (na <= ta and nr <= tr) continue;
            std.debug.print("FAIL {s}: column {d} native abs {e:.3} rel {e:.3} worse than Triton's {e:.3} {e:.3}\n", .{ n.name, c, na, nr, ta, tr });
            failed = true;
            break;
        };
        if (failed) return error.TestFailed;
    }
};

/// Runs a gate so every output's line prints before the run fails: a failure only sets `failed`.
pub fn gate(failed: *bool, result: anyerror!void) !void {
    result catch |e| {
        if (e != error.TestFailed) return e;
        failed.* = true;
    };
}
