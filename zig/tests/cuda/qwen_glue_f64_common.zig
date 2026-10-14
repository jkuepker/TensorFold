//! Shared pieces of the row glue's float64 test: shapes, slacks, an output's statistics per phase, group-sum checks.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const qwen27 = @import("qwen27");
const check = @import("check.zig");
const base = @import("qwen_glue_tests.zig");
const q = @import("qwen_f64.zig");

const nref = core.norms_ref;

pub const glue = qwen27.glue_native;
pub const Rig = base.Rig;
pub const eps: f32 = 1e-6;
pub const hidden = glue.hidden;
pub const vh = glue.value_heads;
pub const dv = glue.value_dim;
pub const heads = glue.query_heads;
pub const hd = glue.head_dim;
pub const vocab = 4096;
pub const kdim = 2048;
// A row whose sum of squares passes this overflows fp32 in every implementation: not a precision case.
pub const overflow: f64 = 1e38;
// Relative slack of a norm (sum tree, sqrt and division approximations, products) and of a sigmoid (ex2, division).
pub const rel_norm: f64 = 0x1p-19;
pub const rel_sigmoid: f64 = 0x1p-20;
// The sigmoid's exp argument is rounded in fp32: a relative error of |z| 2^-24 or so in exp(z), for z below 0.
pub const rel_arg: f64 = 0x1p-22;
// The relative floor upstream's tolerances add (core/norms_ref.zig): a reference below it has no relative error.
pub const floor: f64 = 1e-30;
// fp32 adds on a group sum's deepest path: chain of 8 (7) then butterfly<4> (3); chain of 4 (3), butterfly<8> (4)
pub const depth_row = 10;
pub const depth_head = 7;

pub const Impl = enum { native, triton };

pub fn f(v: u16) f64 {
    return q.bf(v);
}

/// The sigmoid kernels' relative slack at z: the arithmetic's own, and the exp argument's for z below 0.
pub fn sig(rel: f64, z: f64) f64 {
    return rel + (if (z < 0) @min(-z, 100) * rel_arg else 0);
}

pub fn fetch(comptime T: type, r: *Rig, a: std.mem.Allocator, b: cuda.DeviceBuffer, n: usize) ![]T {
    try r.s.synchronize();
    const raw = try check.download(r.gpu, b);
    defer r.gpu.gpa.free(raw);
    const o = try a.alloc(T, n);
    @memcpy(std.mem.sliceAsBytes(o), raw[0 .. n * @sizeOf(T)]);
    return o;
}

/// The row counts one call is checked at: its own and, for a prompt, the first decode-size rows of the same buffers.
pub fn spans(rows: usize, buf: *[2]usize) []const usize {
    buf.* = .{ rows, q.decode_rows };
    return buf[0 .. if (q.phaseOf(rows) == .prompt) 2 else 1];
}

pub fn named(a: std.mem.Allocator, tag: []const u8, what: []const u8) ![]const u8 {
    return std.fmt.allocPrint(a, "{s}.{s}", .{ tag, what });
}

/// One output: its statistics for decode and for prompt row counts apart, and the bound it is held to, as text.
pub const Out = struct {
    p: q.Phased,
    text: []const u8,

    pub fn init(a: std.mem.Allocator, name: []const u8, cols: usize, text: []const u8) !Out {
        return .{ .p = try q.Phased.init(a, name, cols), .text = text };
    }

    pub fn at(o: *Out, rows: usize) *q.Pair {
        return o.p.at(rows);
    }

    /// Prints the phases that ran; counts elements past their bound (Triton's too) and columns worse than Triton.
    pub fn report(o: *const Out) usize {
        var bad: usize = 0;
        for (&o.p.by) |*x| if (x.n.count > 0) {
            x.finish(o.text, true) catch {};
            bad += x.n.fails + x.t.fails + @intFromBool(x.worseCol() != null);
        };
        return bad;
    }

    /// The first phase that ran with its perturbed column and still passed both gates.
    pub fn missed(o: *const Out) ?[]const u8 {
        for (&o.p.by) |*x| if (x.n.count > 0 and x.passes(true)) return x.n.name;
        return null;
    }
};

/// One kernel's outputs: its rows (and the residual sum), and its fused group sums against float64 and its stored rows.
pub const Group = struct {
    tag: []const u8,
    depth: usize,
    h: Out, // H against the host's bf16(fp32 x + r), exactly
    hx: Out, // H against the float64 sum x + r
    y: Out,
    xs: Out,
    sto: Out,
    excluded: usize = 0,

    pub fn init(a: std.mem.Allocator, mode: []const u8, kernel: []const u8, main: []const u8, y_text: []const u8, cols: usize, depth: usize) !Group {
        const tag = try std.fmt.allocPrint(a, "{s}:{s}", .{ mode, kernel });
        const xs_text = try std.fmt.allocPrint(a, "sum of the {s} bounds + {d}u sum|y|, u=2^-24", .{ main, depth });
        const sto_text = try std.fmt.allocPrint(a, "{d}u sum|stored y|, u=2^-24", .{depth});
        return .{
            .tag = tag,
            .depth = depth,
            .h = try Out.init(a, try named(a, tag, "H"), cols, "bf16(fp32(x)+fp32(r)) exactly"),
            .hx = try Out.init(a, try named(a, tag, "H(exact sum)"), cols, "halfUlp(x+r)"),
            .y = try Out.init(a, try named(a, tag, main), cols, y_text),
            .xs = try Out.init(a, try named(a, tag, "XS"), cols / 64, xs_text),
            .sto = try Out.init(a, try named(a, tag, "XS(stored)"), cols / 64, sto_text),
        };
    }

    pub fn all(g: *Group) [5]*Out {
        return .{ &g.h, &g.hx, &g.y, &g.xs, &g.sto };
    }

    pub fn perturb(g: *Group, col: usize) void {
        for (g.all()) |o| o.p.perturb(col);
    }

    pub fn report(g: *const Group) usize {
        var bad: usize = 0;
        for ([_]*const Out{ &g.h, &g.hx, &g.y, &g.xs, &g.sto }) |o| bad += o.report();
        if (g.excluded > 0) std.debug.print("F64 {s} excluded={d} (rows whose sum of squares overflows fp32)\n", .{ g.tag, g.excluded });
        return bad;
    }
};

/// One element whose reference is its implementation's own (a sum of its own stored rows).
pub const El = struct { got: f64, want: f64, limit: f64, floor: f64 };

/// Native, native over the elements Triton produced, and Triton, each against its own reference.
pub fn addOwn(p: *q.Pair, col: usize, n: El, t: ?El) void {
    p.n.add(col, n.got, n.want, n.limit, n.floor);
    if (t) |e| {
        p.np.add(col, n.got, n.want, n.limit, n.floor);
        p.t.add(col, e.got, e.want, e.limit, e.floor);
    }
}

/// An implementation's group sums of one row, and its stored outputs they sum.
pub const Row = struct { xs: []const f32, y: []const f64 };

fn sumAbs(v: []const f64) f64 {
    var t: f64 = 0;
    for (v) |e| t += @abs(e);
    return t;
}

fn ownEl(t: Row, want: f64, j: usize, depth: usize) El {
    const y = t.y[64 * j ..][0..64];
    return .{ .got = t.xs[j], .want = want, .limit = q.sumBound(y, depth), .floor = sumAbs(y) };
}

/// One row's group sums against the float64 rows `want` (bounds `lim` summed plus the reduction's) and the stored ones.
pub fn xsRow(g: *Group, n: usize, col0: usize, nat: Row, tri: ?Row, want: []const f64, lim: []const f64) void {
    var ref: [32]f64 = undefined;
    var own: [2][32]f64 = undefined;
    const groups = nat.xs.len;
    nref.groupSums(want, ref[0..groups]);
    nref.groupSums(nat.y, own[0][0..groups]);
    if (tri) |t| nref.groupSums(t.y, own[1][0..groups]);
    const grow = 1 + @as(f64, @floatFromInt(g.depth)) * q.f32_eps;
    for (nat.xs, 0..) |x, j| {
        var sl: f64 = 0;
        for (lim[64 * j ..][0..64]) |l| sl += l;
        const w = want[64 * j ..][0..64];
        const t_xs: ?f64 = if (tri) |t| t.xs[j] else null;
        g.xs.at(n).add(col0 + j, x, t_xs, ref[j], sl * grow + q.sumBound(w, g.depth), sumAbs(w));
        const sn = nat.y[64 * j ..][0..64];
        const nat_el: El = .{ .got = x, .want = own[0][j], .limit = q.sumBound(sn, g.depth), .floor = sumAbs(sn) };
        const tri_el: ?El = if (tri) |t| ownEl(t, own[1][j], j, g.depth) else null;
        addOwn(g.sto.at(n), col0 + j, nat_el, tri_el);
    }
}

/// Real rows to feed add_rmsnorm: x, and the residual when the call has one.
pub const Real = struct { x: []const u16, r: ?[]const u16 = null };
