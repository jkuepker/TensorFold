//! The gated delta rule in float64 down a chain or tree, each step carrying the fp32 rounding bound it injects.

const std = @import("std");
const f64t = @import("qwen_f64.zig");

pub const dk = 128;
pub const dv = 128;
pub const hk = 16;
pub const hv = 32;
pub const state_len = hv * dv * dk;
// fp32 steps: gated state, a 128-term dot (a few levels), the delta, the update; 32 units of 2^-24 for each
const step_eps: f64 = 32 * 0x1p-24;

/// Every node's q and k [node][key head][dk], v [node][value head][dv] in bf16, g and beta [node][value head].
pub const Inputs = struct { q: []const u16, k: []const u16, v: []const u16, g: []const f32, beta: []const f32 };

pub const NodeOut = struct {
    y: [dv]f64,
    y_bound: [dv]f64, // for the bf16 output
    s_bound: [dv]f64, // for any element of the state's row
    y_floor: [dv]f64,
    s_floor: [dv]f64,
};

fn step(in: *const Inputs, head: usize, node: usize, s_in: []const f64, e_in: *const [dv]f64, s_out: []f64, e_out: *[dv]f64, out: *NodeOut) void {
    const kh = head / (hv / hk);
    var kvec: [dk]f64 = undefined;
    var qvec: [dk]f64 = undefined;
    var qn: f64 = 0;
    for (&kvec, &qvec, 0..) |*a, *b, i| {
        a.* = f64t.bf(in.k[(node * hk + kh) * dk + i]);
        b.* = f64t.bf(in.q[(node * hk + kh) * dk + i]);
        qn += b.* * b.*;
    }
    qn = @sqrt(qn);
    const g: f64 = in.g[node * hv + head];
    const beta: f64 = in.beta[node * hv + head];
    for (0..dv) |row| {
        const si = s_in[row * dk ..][0..dk];
        const so = s_out[row * dk ..][0..dk];
        var mem: f64 = 0;
        var gn: f64 = 0;
        for (si, so, kvec) |x, *o, kk| {
            o.* = x * g;
            mem += o.* * kk;
            gn += o.* * o.*;
        }
        const delta = (f64t.bf(in.v[(node * hv + head) * dv + row]) - mem) * beta;
        var y: f64 = 0;
        var sn: f64 = 0;
        for (so, kvec, qvec) |*o, kk, qq| {
            o.* += kk * delta;
            y += o.* * qq;
            sn += o.* * o.*;
        }
        const sb = g * e_in[row] + step_eps * (@sqrt(gn) + @sqrt(sn));
        e_out[row] = sb;
        out.y[row] = y;
        out.s_bound[row] = sb;
        // the fp32 state's error through q, the dot itself, then the bf16 store (half an ulp, 2^-8 relative at most)
        out.y_bound[row] = qn * (sb + step_eps * @sqrt(sn)) + 0x1p-8 * @abs(y) + 0x1p-133;
        out.y_floor[row] = qn * @sqrt(sn) * 0x1p-7 + 0x1p-133;
        out.s_floor[row] = @sqrt(sn) * 0x1p-7 + 0x1p-133;
    }
}

fn load(dst: []f64, src: []const f32) void {
    for (dst, src) |*d, s| d.* = s;
}

/// Down `nodes` in order from `init` (one head's [dv][dk] state): `ctx.node(head, j, out, state)` after each.
pub fn chain(gpa: std.mem.Allocator, ctx: anytype, in: *const Inputs, head: usize, nodes: []const usize, init: []const f32) !void {
    const bufs = try gpa.alloc(f64, 2 * dv * dk);
    defer gpa.free(bufs);
    var errs: [2][dv]f64 = .{ @splat(0), @splat(0) };
    load(bufs[0 .. dv * dk], init);
    var out: NodeOut = undefined;
    var at: usize = 0;
    for (nodes, 0..) |node, j| {
        const next = 1 - at;
        step(in, head, node, bufs[at * dv * dk ..][0 .. dv * dk], &errs[at], bufs[next * dv * dk ..][0 .. dv * dk], &errs[next], &out);
        at = next;
        ctx.node(head, j, &out, bufs[at * dv * dk ..][0 .. dv * dk]);
    }
}

/// A window of `parents.len` nodes starting at `first`: each node steps from its parent's state (-1: `init`).
pub fn tree(gpa: std.mem.Allocator, ctx: anytype, in: *const Inputs, head: usize, first: usize, parents: []const i32, init: []const f32) !void {
    const n = parents.len;
    const states = try gpa.alloc(f64, n * dv * dk);
    defer gpa.free(states);
    const errs = try gpa.alloc([dv]f64, n);
    defer gpa.free(errs);
    const base = try gpa.alloc(f64, dv * dk);
    defer gpa.free(base);
    load(base, init);
    const zero: [dv]f64 = @splat(0);
    var out: NodeOut = undefined;
    for (parents, 0..) |p, j| {
        const s_in = if (p < 0) base else states[@as(usize, @intCast(p)) * dv * dk ..][0 .. dv * dk];
        const e_in = if (p < 0) &zero else &errs[@intCast(p)];
        step(in, head, first + j, s_in, e_in, states[j * dv * dk ..][0 .. dv * dk], &errs[j], &out);
        ctx.node(head, j, &out, states[j * dv * dk ..][0 .. dv * dk]);
    }
}
