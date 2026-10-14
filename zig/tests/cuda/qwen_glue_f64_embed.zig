//! The embedding rows of the row glue's float64 test: a 4-bit affine table and its dequantized rows against float64.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const base = @import("qwen_glue_tests.zig");
const q = @import("qwen_f64.zig");
const c = @import("qwen_glue_f64_common.zig");

const aref = core.act_ref;
const Rig = c.Rig;
const kdim = c.kdim;

/// The 4-bit affine embedding table of the differential test: random words, scales near 0.05, biases within 1.
pub const Table = struct {
    words: []u32,
    sc: []u16,
    bi: []u16,
    dev: [3]cuda.DeviceBuffer,

    pub fn init(r: *Rig, a: std.mem.Allocator) !Table {
        const rng = r.rand();
        var t: Table = .{
            .words = try a.alloc(u32, c.vocab * kdim / 8),
            .sc = try a.alloc(u16, c.vocab * kdim / 64),
            .bi = try a.alloc(u16, c.vocab * kdim / 64),
            .dev = undefined,
        };
        for (t.words) |*w| w.* = rng.int(u32);
        for (t.sc) |*o| o.* = base.bf((rng.float(f32) + 0.05) * 0.1);
        for (t.bi) |*o| o.* = base.bf((rng.float(f32) - 0.5) * 2.0);
        t.dev[0] = try cuda.DeviceBuffer.fromHost(r.gpu.d, std.mem.sliceAsBytes(t.words));
        t.dev[1] = try cuda.DeviceBuffer.fromHost(r.gpu.d, base.bytes(t.sc));
        t.dev[2] = try cuda.DeviceBuffer.fromHost(r.gpu.d, base.bytes(t.bi));
        return t;
    }

    pub fn deinit(t: *Table) void {
        for (&t.dev) |*b| b.free();
    }
};

/// embed: q scale + bias in float64 from the same words, scales and biases, for random token rows.
pub fn embed(r: *Rig, o: *c.Out, t: Table, rows: usize, tri: bool) !void {
    var arena = std.heap.ArenaAllocator.init(r.gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    defer r.release();
    const rng = r.rand();
    const ids = try a.alloc(i32, rows);
    for (ids) |*x| x.* = @intCast(rng.uintLessThan(u32, c.vocab));
    ids[0] = if (rng.boolean()) 0 else c.vocab - 1;
    const ib = try r.up(std.mem.sliceAsBytes(ids), true);
    var sp: [2]usize = undefined;
    for (c.spans(rows, &sp)) |n| {
        const on = try r.out(n * kdim * 2);
        try r.native.embed(r.s, ib.ptr, t.dev[0].ptr, t.dev[1].ptr, t.dev[2].ptr, on.ptr, n, kdim);
        const nat = try c.fetch(u16, r, a, on, n * kdim);
        var other: ?[]u16 = null;
        if (tri) {
            const ot = try r.out(n * kdim * 2);
            try r.replay.embed(ib.ptr, t.dev[0].ptr, t.dev[1].ptr, t.dev[2].ptr, ot.ptr, n, kdim);
            other = try c.fetch(u16, r, a, ot, n * kdim);
        }
        for (ids[0..n], 0..) |id, row| {
            const tok: usize = @intCast(id);
            for (0..kdim) |i| {
                const word = t.words[tok * (kdim / 8) + i / 8];
                const qv: u4 = @intCast((word >> @intCast(4 * (i % 8))) & 0xf);
                const want = aref.affine4(qv, c.f(t.sc[tok * (kdim / 64) + i / 64]), c.f(t.bi[tok * (kdim / 64) + i / 64]));
                const got_t: ?f64 = if (other) |x| c.f(x[row * kdim + i]) else null;
                o.at(n).add(i, c.f(nat[row * kdim + i]), got_t, want, q.bound(want, 0x1p-23, 0), c.floor);
            }
        }
    }
}
