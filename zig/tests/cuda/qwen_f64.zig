//! Float64 reference scaffolding shared by the Qwen3.6 precision tests: bf16 decode, ulp spacing, error statistics.

const std = @import("std");

pub fn bf(x: u16) f64 {
    return @as(f32, @bitCast(@as(u32, x) << 16));
}

/// fp32 to bf16, round to nearest even, as the kernels' own rounding; NaN stays a NaN.
pub fn bf16Of(v: f32) u16 {
    const u: u32 = @bitCast(v);
    if (std.math.isNan(v)) return 0x7fc0;
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// fp32 to bf16 by truncation (round toward zero): what a kernel rounding with the wrong mode would store.
pub fn bf16Trunc(v: f32) u16 {
    return @intCast(@as(u32, @bitCast(v)) >> 16);
}

/// A float64 value rounded to bf16 through fp32 and back, as the kernels round.
pub fn round(x: f64) f64 {
    return bf(bf16Of(@floatCast(x)));
}

pub fn bfAt(raw: []const u8, i: usize) f64 {
    return bf(std.mem.readInt(u16, raw[2 * i ..][0..2], .little));
}

pub fn f32At(raw: []const u8, i: usize) f64 {
    return @as(f32, @bitCast(std.mem.readInt(u32, raw[4 * i ..][0..4], .little)));
}

pub fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

/// The spacing of bf16 values around `x` (8 significand bits), at least the smallest subnormal step.
pub fn ulpBf16(x: f64) f64 {
    const a = @abs(x);
    if (!(a > 0) or !std.math.isFinite(a)) return 0x1p-133;
    return @max(@exp2(@as(f64, @floatFromInt(std.math.frexp(a).exponent - 8))), 0x1p-133);
}

/// Half a bf16 ulp at |w| (the subnormal spacing below 2^-126): the most a once-rounded output can sit from its value.
pub fn halfUlp(w: f64) f64 {
    const bits: u64 = @bitCast(@abs(w));
    const e: i32 = @max(@as(i32, @intCast(bits >> 52)) - 1023, -126);
    return @bitCast(@as(u64, @intCast(e - 8 + 1023)) << 52);
}

pub const flt_min: f64 = 0x1p-126;
/// Two fp32 subnormal steps: below 2^-126 an fp32 result keeps fewer bits, an error a relative slack cannot cover.
pub const subnormal: f64 = 0x1p-148;
/// Half a bf16 ulp at the top of a binade, relative: what rounding an output to bf16 can cost.
pub const eps_bf16: f64 = 0x1p-8;
pub const f32_eps: f64 = 0x1p-24; // the fp32 unit roundoff

/// A bf16 output's bound: its rounding plus `rel` of |w| plus `abs` plus the fp32 subnormal step.
pub fn bound(w: f64, rel: f64, abs: f64) f64 {
    return halfUlp(w) + @abs(w) * rel + abs + subnormal;
}

/// An fp32 sum's error bound when no input passes through more than `depth` roundings: depth u sum |x|.
pub fn sumBound(x: []const f64, depth: usize) f64 {
    var t: f64 = 0;
    for (x) |v| t += @abs(v);
    return t * @as(f64, @floatFromInt(depth)) * f32_eps + 1e-30;
}

/// What a row count is: a decode step, lane round or verify window (up to 16 rows), or a prompt chunk.
pub const Phase = enum(u1) { decode, prompt };
pub const decode_rows = 16;

pub fn phaseOf(rows: usize) Phase {
    return if (rows <= decode_rows) .decode else .prompt;
}

pub const Kind = enum { bf16, f32 };

/// `n` values of bf16 or fp32 words at `bytes`, as float64.
pub fn decode(a: std.mem.Allocator, bytes: []const u8, kind: Kind, n: usize) ![]f64 {
    const out = try a.alloc(f64, n);
    for (out, 0..) |*v, i| v.* = switch (kind) {
        .bf16 => bfAt(bytes, i),
        .f32 => f32At(bytes, i),
    };
    return out;
}

/// The column whose elements the negative control pushes off the reference (TF_QWEN_F64_PERTURB=<column>).
pub fn controlColumn() ?usize {
    const v = std.c.getenv("TF_QWEN_F64_PERTURB") orelse return null;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch null;
}

/// One output's errors against its float64 reference, per column and over all elements.
pub const Stat = struct {
    gpa: std.mem.Allocator,
    name: []const u8,
    abs: []f64, // per column: worst absolute error
    rel: []f64, // per column: worst error over max(|want|, floor)
    ratio: []f64, // per column: worst error over its bound
    bound: []f64, // per column: largest bound
    inject: bool, // the negative control may move this statistic's elements (the native one only)
    control: ?usize,
    count: usize = 0,
    fails: usize = 0, // elements past their bound (NaN counts)
    sum_abs: f64 = 0, // for the aggregate mean absolute error beside the worst column
    sum_sq: f64 = 0, // and the RMS
    sum_ratio: f64 = 0,
    first: ?struct { col: usize, got: f64, want: f64, bound: f64 } = null, // the first element past its bound

    pub fn init(gpa: std.mem.Allocator, name: []const u8, cols: usize, inject: bool) !Stat {
        const all = try gpa.alloc(f64, 4 * cols);
        @memset(all, 0);
        return .{ .gpa = gpa, .name = name, .abs = all[0..cols], .rel = all[cols .. 2 * cols], .ratio = all[2 * cols .. 3 * cols], .bound = all[3 * cols ..], .inject = inject, .control = controlColumn() };
    }

    pub fn deinit(s: *Stat) void {
        s.gpa.free(s.abs.ptr[0 .. 4 * s.abs.len]);
    }

    /// The negative control: every element of `col` this statistic sees moves 1.25 bounds off the reference.
    pub fn perturb(s: *Stat, col: usize) void {
        s.inject = true;
        s.control = col;
    }

    /// One element of column `col`: `got` against `want` within `bound`; relative error is over max(|want|, floor).
    pub fn add(s: *Stat, col: usize, got: f64, want: f64, limit: f64, floor: f64) void {
        var g = got;
        const b = @max(limit, 1e-300);
        if (s.inject and s.control != null and s.control.? == col) g += if (g >= want) 1.25 * b else -1.25 * b;
        const err = if (std.math.isFinite(g)) @abs(g - want) else std.math.inf(f64);
        s.count += 1;
        s.sum_abs += err;
        s.sum_sq += err * err;
        s.sum_ratio += err / b;
        // NaN compares false: an unwritten or poisoned value is the worst
        if (!(err <= b)) {
            s.fails += 1;
            if (s.first == null) s.first = .{ .col = col, .got = g, .want = want, .bound = b };
        }
        s.abs[col] = @max(s.abs[col], err);
        s.rel[col] = @max(s.rel[col], err / @max(@abs(want), floor));
        s.ratio[col] = @max(s.ratio[col], err / b);
        s.bound[col] = @max(s.bound[col], b);
    }

    /// `got` against `want`, `tol` (> 0), `cols` to a row; relative error is over max(|want|, the column peak / 1024).
    pub fn addSlices(s: *Stat, got: []const f64, want: []const f64, tol: []const f64) !void {
        const cols = s.abs.len;
        const peak = try s.gpa.alloc(f64, cols);
        defer s.gpa.free(peak);
        @memset(peak, 0);
        for (want, 0..) |w, i| peak[i % cols] = @max(peak[i % cols], @abs(w));
        for (got, want, tol, 0..) |g, w, t, i| s.add(i % cols, g, w, t, @max(peak[i % cols] * 0x1p-10, std.math.floatMin(f32)));
    }

    /// Folds `o` (the same columns) into `s`: worst per column, sums added.
    pub fn merge(s: *Stat, o: *const Stat) void {
        for (s.abs, s.rel, s.ratio, s.bound, 0..) |*a, *r, *q, *b, i| {
            a.* = @max(a.*, o.abs[i]);
            r.* = @max(r.*, o.rel[i]);
            q.* = @max(q.*, o.ratio[i]);
            b.* = @max(b.*, o.bound[i]);
        }
        s.count += o.count;
        s.fails += o.fails;
        s.sum_abs += o.sum_abs;
        s.sum_sq += o.sum_sq;
        s.sum_ratio += o.sum_ratio;
        if (s.first == null) s.first = o.first;
    }

    pub fn meanAbs(s: *const Stat) f64 {
        return s.sum_abs / @as(f64, @floatFromInt(@max(s.count, 1)));
    }

    pub fn rms(s: *const Stat) f64 {
        return @sqrt(s.sum_sq / @as(f64, @floatFromInt(@max(s.count, 1))));
    }

    /// The worst column's value and index (the first on ties; a NaN counts as worse than any number).
    pub fn worst(v: []const f64) struct { at: usize, val: f64 } {
        var at: usize = 0;
        for (v, 0..) |x, i| if (!(x <= v[at])) {
            at = i;
        };
        return .{ .at = at, .val = v[at] };
    }

    pub fn print(s: *const Stat, label: []const u8) void {
        const a = worst(s.abs);
        const r = worst(s.rel);
        const q = worst(s.ratio);
        const bd = worst(s.bound);
        std.debug.print(" {s} abs={e:.3} (col {d}) rel={e:.3} (col {d}) mean_abs={e:.3} rms={e:.3} err/bound={d:.3} (col {d}) mean_err/bound={d:.4} bound<={e:.3} n={d}", .{ label, a.val, a.at, r.val, r.at, s.meanAbs(), s.rms(), q.val, q.at, s.sum_ratio / @as(f64, @floatFromInt(@max(s.count, 1))), bd.val, s.count });
    }
};

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

    pub fn perturb(p: *Pair, col: usize) void {
        p.n.perturb(col);
        p.np.perturb(col);
    }

    pub fn add(p: *Pair, col: usize, got: f64, tri_got: ?f64, want: f64, limit: f64, floor: f64) void {
        p.n.add(col, got, want, limit, floor);
        if (tri_got) |g| {
            p.np.add(col, got, want, limit, floor);
            p.t.add(col, g, want, limit, floor);
        }
    }

    /// The first column where native, over the elements Triton also produced, is worse than Triton.
    pub fn worseCol(p: *const Pair) ?usize {
        if (p.t.count == 0) return null;
        for (p.np.abs, p.t.abs, p.np.rel, p.t.rel, 0..) |na, ta, nr, tr, c| {
            if (!(na <= ta and nr <= tr)) return c;
        }
        return null;
    }

    /// Whether the output meets its gates: every column within its bound (when `strict`), native no worse than Triton.
    pub fn passes(p: *const Pair, strict: bool) bool {
        return !(strict and p.n.fails > 0) and p.worseCol() == null;
    }

    /// One machine-readable line (native, then Triton), then both gates.
    pub fn finish(p: *const Pair, bound_text: []const u8, strict: bool) !void {
        const n = &p.n;
        std.debug.print("F64 {s} cols={d} native", .{ n.name, n.abs.len });
        n.print("");
        if (p.t.count > 0) p.t.print(" triton") else std.debug.print(" triton none", .{});
        std.debug.print(" over={d} bound=\"{s}\" strict={}\n", .{ n.fails, bound_text, strict });
        var failed = false;
        if (strict and n.fails > 0) {
            const f = n.first.?;
            std.debug.print("FAIL {s}: {d} elements past their bound, first in column {d}: got {e:.6}, want {e:.6}, bound {e:.3}\n", .{ n.name, n.fails, f.col, f.got, f.want, f.bound });
            failed = true;
        }
        if (p.worseCol()) |c| {
            std.debug.print("FAIL {s}: column {d} native abs {e:.3} rel {e:.3} worse than Triton's {e:.3} {e:.3}\n", .{ n.name, c, p.np.abs[c], p.np.rel[c], p.t.abs[c], p.t.rel[c] });
            failed = true;
        }
        if (failed) return error.TestFailed;
    }
};

/// A Pair per phase, the decode and prompt row counts reported apart: `name.decode`, `name.prompt`.
pub const Phased = struct {
    by: [2]Pair,

    pub fn init(gpa: std.mem.Allocator, name: []const u8, cols: usize) !Phased {
        var p: Phased = undefined;
        for (&p.by, 0..) |*x, i| x.* = try Pair.init(gpa, try std.fmt.allocPrint(gpa, "{s}.{t}", .{ name, @as(Phase, @enumFromInt(i)) }), cols);
        return p;
    }

    pub fn deinit(p: *Phased) void {
        for (&p.by) |*x| {
            x.n.gpa.free(x.n.name);
            x.deinit();
        }
    }

    pub fn at(p: *Phased, rows: usize) *Pair {
        return &p.by[@intFromEnum(phaseOf(rows))];
    }

    pub fn perturb(p: *Phased, col: usize) void {
        for (&p.by) |*x| x.perturb(col);
    }

    /// Both phases' lines (a phase nothing ran in prints nothing), then the first failure.
    pub fn finish(p: *const Phased, bound_text: []const u8, strict: bool) !void {
        var failed = false;
        for (&p.by) |*x| if (x.n.count > 0) try gate(&failed, x.finish(bound_text, strict));
        if (failed) return error.TestFailed;
    }

    pub fn passes(p: *const Phased, strict: bool) bool {
        for (&p.by) |*x| if (!x.passes(strict)) return false;
        return true;
    }
};

/// Runs a gate so every output's line prints before the run fails: a failure only sets `failed`.
pub fn gate(failed: *bool, result: anyerror!void) !void {
    result catch |e| {
        if (e != error.TestFailed) return e;
        failed.* = true;
    };
}

test "aggregates beside the worst column" {
    var s = try Stat.init(std.testing.allocator, "x", 2, false);
    defer s.deinit();
    s.add(0, 1.0, 1.0, 1, 1);
    s.add(1, 1.5, 1.0, 1, 1);
    s.add(1, 1.0, 2.0, 2, 1);
    try std.testing.expectEqual(@as(usize, 3), s.count);
    try std.testing.expectEqual(@as(f64, 0), s.abs[0]);
    try std.testing.expectEqual(@as(f64, 1), s.abs[1]);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), s.rel[1], 1e-15);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), s.meanAbs(), 1e-15);
    try std.testing.expectApproxEqAbs(@sqrt(@as(f64, 1.25 / 3.0)), s.rms(), 1e-15);
    try std.testing.expectEqual(@as(usize, 0), s.fails);
    s.add(0, 3.0, 1.0, 1, 1); // two over a bound of 1
    try std.testing.expectEqual(@as(usize, 1), s.fails);
    try std.testing.expectEqual(@as(usize, 0), s.first.?.col);
}

test "a NaN is the worst and is counted" {
    var s = try Stat.init(std.testing.allocator, "x", 3, false);
    defer s.deinit();
    s.add(2, std.math.nan(f64), 1.0, 1, 1);
    s.add(1, 1.0, 1.0, 1, 1);
    try std.testing.expectEqual(@as(usize, 1), s.fails);
    try std.testing.expect(std.math.isInf(s.abs[2]));
    try std.testing.expectEqual(@as(usize, 2), Stat.worst(s.abs).at);
}

test "native worse than Triton in one column fails the pair, a clean one passes" {
    var p = try Pair.init(std.testing.allocator, "x", 2);
    defer p.deinit();
    p.add(0, 1.0, 1.0, 1.0, 1, 1);
    p.add(1, 1.25, 1.0, 1.0, 1, 1);
    try std.testing.expectEqual(@as(?usize, 1), p.worseCol());
    try std.testing.expect(!p.passes(true));
    var q = try Pair.init(std.testing.allocator, "y", 1);
    defer q.deinit();
    q.add(0, 1.25, null, 1.0, 1, 1);
    try std.testing.expect(q.passes(true));
}

test "a perturbed column fails and the others do not" {
    var s = try Stat.init(std.testing.allocator, "x", 2, true);
    defer s.deinit();
    s.perturb(1);
    s.add(0, 1.0, 1.0, 1, 1);
    s.add(1, 1.0, 1.0, 1, 1);
    try std.testing.expectEqual(@as(usize, 1), s.fails);
    try std.testing.expectEqual(@as(usize, 1), s.first.?.col);
}

test "phases split at 16 rows" {
    try std.testing.expectEqual(Phase.decode, phaseOf(1));
    try std.testing.expectEqual(Phase.decode, phaseOf(16));
    try std.testing.expectEqual(Phase.prompt, phaseOf(17));
    var p = try Phased.init(std.testing.allocator, "k", 1);
    defer p.deinit();
    p.at(4).n.add(0, 1.0, 1.0, 1, 1);
    p.at(299).n.add(0, 2.0, 1.0, 1, 1);
    try std.testing.expectEqual(@as(usize, 1), p.by[0].n.count);
    try std.testing.expectEqual(@as(f64, 1), p.by[1].n.abs[0]);
    try std.testing.expectEqual(@as(f64, 0), p.by[0].n.abs[0]);
}

test "round to nearest even and toward zero differ where a kernel's rounding mode would" {
    const x: f32 = @bitCast(@as(u32, 0x3f80_8001)); // just past the midpoint between two bf16 values
    try std.testing.expectEqual(@as(u16, 0x3f81), bf16Of(x));
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16Trunc(x));
    const tie: f32 = @bitCast(@as(u32, 0x3f80_8000)); // a tie goes to the even mantissa
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16Of(tie));
    try std.testing.expectEqual(@as(u16, 0x7fc0), bf16Of(std.math.nan(f32)));
    try std.testing.expectEqual(@as(f64, 1.0078125), round(1.0078125 + 0x1p-12));
}

test "half ulps" {
    try std.testing.expectEqual(@as(f64, 0x1p-8), halfUlp(1.0));
    try std.testing.expectEqual(@as(f64, 0x1p-8), halfUlp(1.99));
    try std.testing.expectEqual(@as(f64, 0x1p-7), halfUlp(2.0));
    try std.testing.expectEqual(@as(f64, 0x1p-134), halfUlp(0));
    try std.testing.expectEqual(@as(f64, 0x1p-134), halfUlp(0x1p-130));
}

test "a depth-d reduction bound is d units of 2^-24 of the absolute sum" {
    const x = [_]f64{ 1, -1, 2 };
    try std.testing.expectApproxEqRel(4.0 * 10 * 0x1p-24, sumBound(&x, 10), 1e-12);
    try std.testing.expect(sumBound(&x, 7) < sumBound(&x, 10));
}
