//! Error accounting for the row glue's float64 checks: worst per-column errors, once-rounded bounds, the printed table.
const std = @import("std");

/// The relative floor upstream's tolerances add (core/norms_ref.zig): a reference below it has no relative error.
pub const floor: f64 = 1e-30;
/// The smallest normal fp32: the fp32 sigmoid's division flushes results below it.
pub const flush: f64 = 0x1p-126;
/// Half a bf16 ulp at |w| (the subnormal spacing below 2^-126): the most a once-rounded output can sit from its value.
pub fn halfUlp(w: f64) f64 {
    const bits: u64 = @bitCast(@abs(w));
    const e: i32 = @max(@as(i32, @intCast(bits >> 52)) - 1023, -126);
    return @bitCast(@as(u64, @intCast(e - 8 + 1023)) << 52);
}

/// Two fp32 subnormal steps: below 2^-126 an fp32 result keeps fewer bits, an error a relative slack cannot cover.
pub const subnormal: f64 = 0x1p-148;

/// A bf16 output's bound: its rounding plus `rel` of |w| plus `abs` plus the fp32 subnormal step.
pub fn bound(w: f64, rel: f64, abs: f64) f64 {
    return halfUlp(w) + @abs(w) * rel + abs + subnormal;
}

pub const Impl = enum(u1) { native, triton };

/// One implementation's worst errors per column over every row, plus how often a bound broke.
pub const Acc = struct {
    abs: []f64,
    rel: []f64,
    ratio: f64 = 0,
    fails: usize = 0,
    count: usize = 0,
};

/// One output of one kernel: columns are the output's index within a row (hidden index, head and dim, group, d).
pub const Out = struct {
    name: []const u8,
    acc: [2]Acc,
    excluded: usize = 0,
    shown: usize = 0,
    /// The negative control: the first native element at this column from this row on is moved three bounds off.
    poke: ?struct { row: usize, col: usize } = null,

    pub fn init(a: std.mem.Allocator, name: []const u8, cols: usize) !Out {
        var o: Out = .{ .name = name, .acc = undefined };
        for (&o.acc) |*c| {
            c.* = .{ .abs = try a.alloc(f64, cols), .rel = try a.alloc(f64, cols) };
            @memset(c.abs, 0);
            @memset(c.rel, 0);
        }
        return o;
    }

    pub fn deinit(o: *Out, a: std.mem.Allocator) void {
        for (o.acc) |c| {
            a.free(c.abs);
            a.free(c.rel);
        }
    }

    /// One element: its error beside the bound (relative to `scale`, |want| unless a sum's terms); over it, it counts.
    pub fn note(o: *Out, im: Impl, row: usize, col: usize, got_in: f64, want: f64, limit: f64, scale: f64) void {
        const c = &o.acc[@backingInt(im)];
        var got = got_in;
        if (o.poke) |p| if (im == .native and row >= p.row and col == p.col) {
            got = want + 3 * limit;
            o.poke = null;
        };
        const err = @abs(got - want);
        c.count += 1;
        const ok = std.math.isFinite(got) and err <= limit;
        if (std.math.isFinite(err)) {
            c.abs[col] = @max(c.abs[col], err);
            c.rel[col] = @max(c.rel[col], err / @max(scale, floor));
            c.ratio = @max(c.ratio, err / @max(limit, 1e-300));
        }
        if (ok) return;
        c.fails += 1;
        if (o.shown < 6) {
            o.shown += 1;
            std.debug.print("  OVER {s} {t} row {d} col {d}: got {e} want {e} bound {e}\n", .{ o.name, im, row, col, got, want, limit });
        }
    }

    /// The columns where native is worse than Triton, by absolute or relative worst error.
    pub fn worse(o: *const Out) usize {
        var n: usize = 0;
        const a = o.acc[0];
        const t = o.acc[1];
        for (a.abs, a.rel, t.abs, t.rel) |na, nr, ta, tr| {
            if (na > ta or nr > tr) n += 1;
        }
        return n;
    }

    /// The negative control from `row` and `col` on, quietly.
    pub fn arm(o: *Out, row: usize, col: usize) void {
        o.poke = .{ .row = row, .col = col };
        o.shown = 6;
    }

    /// How many things failed: elements over their bound, columns where native is worse than Triton.
    pub fn failures(o: *const Out, triton: bool) usize {
        return o.acc[0].fails + (if (triton) o.acc[1].fails + o.worse() else 0);
    }

    /// The table line, and the failures.
    pub fn report(o: *const Out, triton: bool) usize {
        const n = o.acc[0];
        var wa: usize = 0;
        var wr: usize = 0;
        for (n.abs, n.rel, 0..) |a, r, i| {
            if (a > n.abs[wa]) wa = i;
            if (r > n.rel[wr]) wr = i;
        }
        std.debug.print("F64 {s} cols={d} n={d} excluded={d} | native abs={e}@{d} rel={e}@{d} bound={d:.3} over={d}", .{
            o.name, n.abs.len, n.count, o.excluded, n.abs[wa], wa, n.rel[wr], wr, n.ratio, n.fails,
        });
        if (triton) {
            const t = o.acc[1];
            var ta: f64 = 0;
            var tr: f64 = 0;
            for (t.abs, t.rel) |a, r| {
                ta = @max(ta, a);
                tr = @max(tr, r);
            }
            const w = o.worse();
            std.debug.print(" | triton abs={e} rel={e} bound={d:.3} over={d} | native worse cols={d}\n", .{ ta, tr, t.ratio, t.fails, w });
        } else std.debug.print(" | triton none\n", .{});
        return o.failures(triton);
    }
};

test "half ulps" {
    try std.testing.expectEqual(@as(f64, 0x1p-8), halfUlp(1.0));
    try std.testing.expectEqual(@as(f64, 0x1p-8), halfUlp(1.99));
    try std.testing.expectEqual(@as(f64, 0x1p-7), halfUlp(2.0));
    try std.testing.expectEqual(@as(f64, 0x1p-134), halfUlp(0));
    try std.testing.expectEqual(@as(f64, 0x1p-134), halfUlp(0x1p-130));
}
