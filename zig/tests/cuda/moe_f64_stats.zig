//! Float64 error reports for the Qwen3.6 MoE tests: worst per-column absolute and relative error against a bound.
const std = @import("std");
const check = @import("check.zig");

/// One output's error against its float64 reference; `frac` is the worst difference as a fraction of its bound.
pub const Err = struct {
    abs: f64 = 0,
    abs_col: usize = 0,
    rel: f64 = 0,
    rel_col: usize = 0,
    frac: f64 = 0,
    frac_col: usize = 0,
    at: [4]f64 = .{ 0, 0, 0, 0 }, // the worst element: index, got, want, bound
    count: usize = 0,
};

pub const eps_bf16: f64 = 0x1p-8; // half a bf16 ulp at the top of a binade: what rounding an output to bf16 can cost

pub fn bf(x: u16) f64 {
    return @as(f32, @bitCast(@as(u32, x) << 16));
}

pub fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// A float64 value rounded to bf16 through fp32, as the kernels round.
pub fn round(x: f64) f64 {
    return bf(bf16(@floatCast(x)));
}

pub fn sigmoid(x: f64) f64 {
    return 1 / (1 + @exp(-x));
}

pub const Kind = enum { bf16, f32 };

/// `n` values of bf16 or fp32 words at `bytes`, as float64.
pub fn decode(a: std.mem.Allocator, bytes: []const u8, kind: Kind, n: usize) ![]f64 {
    const out = try a.alloc(f64, n);
    for (out, 0..) |*v, i| v.* = switch (kind) {
        .bf16 => bf(std.mem.bytesToValue(u16, bytes[2 * i ..][0..2])),
        .f32 => std.mem.bytesToValue(f32, bytes[4 * i ..][0..4]),
    };
    return out;
}

/// Errors of `got` against `want` per column (`cols` a row), `tol` > 0; rel is against max(|want|, column peak / 1024).
pub fn measure(a: std.mem.Allocator, got: []const f64, want: []const f64, tol: []const f64, cols: usize) !Err {
    const peak = try a.alloc(f64, cols);
    defer a.free(peak);
    @memset(peak, 0);
    for (want, 0..) |w, i| peak[i % cols] = @max(peak[i % cols], @abs(w));
    var e: Err = .{ .count = want.len };
    for (got, want, tol, 0..) |g, w, t, i| {
        const c = i % cols;
        const d = @abs(g - w);
        const floor = peak[c] * 0x1p-10;
        const rel = d / @max(@abs(w), floor, std.math.floatMin(f32));
        const q = d / t;
        // NaN compares false: an unwritten or poisoned value is the worst
        if (!(d <= e.abs)) {
            e.abs = d;
            e.abs_col = c;
        }
        if (!(rel <= e.rel)) {
            e.rel = rel;
            e.rel_col = c;
        }
        if (!(q <= e.frac)) {
            e.frac = q;
            e.frac_col = c;
            e.at = .{ @floatFromInt(i), g, w, t };
        }
    }
    return e;
}

/// The worse of two errors, column by column at the worst.
pub fn worse(a: Err, b: Err) Err {
    var e = a;
    e.count += b.count;
    if (!(b.abs <= a.abs)) {
        e.abs = b.abs;
        e.abs_col = b.abs_col;
    }
    if (!(b.rel <= a.rel)) {
        e.rel = b.rel;
        e.rel_col = b.rel_col;
    }
    if (!(b.frac <= a.frac)) {
        e.frac = b.frac;
        e.frac_col = b.frac_col;
    }
    return e;
}

/// One summary row: an output's worst errors over a whole run, native and (where there is one) the reference engine's.
pub fn table(name: []const u8, native: Err, reference: ?Err, bound: []const u8) void {
    std.debug.print("TABLE {s}: native abs {e:.3} (col {d}) rel {e:.3} (col {d})", .{ name, native.abs, native.abs_col, native.rel, native.rel_col });
    if (reference) |t| std.debug.print("; triton abs {e:.3} (col {d}) rel {e:.3} (col {d})", .{ t.abs, t.abs_col, t.rel, t.rel_col });
    std.debug.print("; bound {s}, worst {d:.3} of it\n", .{ bound, native.frac });
}

pub fn line(name: []const u8, e: Err) void {
    std.debug.print("RESULT {s}: worst abs {e:.3} (col {d}), worst rel {e:.3} (col {d}), {d:.3} of bound, {d} values\n", .{ name, e.abs, e.abs_col, e.rel, e.rel_col, e.frac, e.count });
}

/// Print the error and require it within the bound.
pub fn ok(name: []const u8, e: Err) !void {
    line(name, e);
    try check.expect(e.frac <= 1, "{s} within its float64 bound (worst {d:.3} of it, element {d:.0}: got {e:.6}, want {e:.6}, bound {e:.2})", .{ name, e.frac, e.at[0], e.at[1], e.at[2], e.at[3] });
}

/// The native error is no worse than the reference engine's, column for column at the worst.
pub fn noWorse(name: []const u8, native: Err, reference: Err) !void {
    try check.expect(native.abs <= reference.abs and native.rel <= reference.rel, "{s}: native abs {e:.3} rel {e:.3} vs reference abs {e:.3} rel {e:.3}", .{ name, native.abs, native.rel, reference.abs, reference.rel });
}

/// Negative control: the value of the largest reference, moved 4 bounds away, must be refused.
pub fn reject(a: std.mem.Allocator, name: []const u8, got: []const f64, want: []const f64, tol: []const f64, cols: usize) !void {
    var at: usize = 0;
    for (want, 0..) |w, i| {
        if (@abs(w) > @abs(want[at])) at = i;
    }
    const bad = try a.dupe(f64, got);
    defer a.free(bad);
    bad[at] += 4 * tol[at];
    const e = try measure(a, bad, want, tol, cols);
    std.debug.print("CONTROL {s}: element {d} moved 4 bounds -> {d:.3} of bound, {s}\n", .{ name, at, e.frac, if (e.frac > 1) "refused" else "ACCEPTED" });
    try check.expect(e.frac > 1, "{s}: a perturbed output is refused", .{name});
}
