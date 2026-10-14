//! The Python engine's dumps as rows for the row glue's float64 test: real y, h, and x and residual rows.
const std = @import("std");
const c = @import("qwen_glue_f64_common.zig");
const ref = @import("qwen_glue_f64_ref.zig");

const kdim = c.kdim;
const prompts = [_][]const u8{ "story", "code", "facts", "long" };
const phases = [_][]const u8{ "cold", "warm", "prefill" };

fn read(comptime T: type, r: *c.Rig, a: std.mem.Allocator, dir: []const u8, comptime fmt: []const u8, args: anytype) !?[]T {
    const path = try std.fmt.allocPrint(a, "{s}/" ++ fmt ++ ".bin", .{dir} ++ args);
    const raw = std.Io.Dir.cwd().readFileAlloc(r.gpu.io, path, a, .limited(1 << 30)) catch |e| switch (e) {
        error.FileNotFound => return null,
        else => return e,
    };
    if (raw.len % @sizeOf(T) != 0) return error.UnexpectedDump;
    const out = try a.alloc(T, raw.len / @sizeOf(T));
    @memcpy(std.mem.sliceAsBytes(out), raw);
    return out;
}

fn present(v: ?[]u16) ![]u16 {
    return v orelse error.FileNotFound;
}

/// Feeds `v` every y file (y), every eighth file's h (add, x alone) and every fourth input norm's h and MoE output.
pub fn walk(r: *c.Rig, oracle: []const u8, v: anytype) !void {
    var files: usize = 0;
    var norms: usize = 0;
    var fed: usize = 0;
    var equal: usize = 0;
    var arena = std.heap.ArenaAllocator.init(r.gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for (prompts) |prompt| for (phases) |phase| {
        var step: usize = 0;
        while (step < 4096) : (step += 1) {
            var dir_buf: [256]u8 = undefined;
            const dir = if (std.mem.eql(u8, phase, "prefill"))
                try std.fmt.bufPrint(&dir_buf, "{s}/{s}/{s}", .{ oracle, prompt, phase })
            else
                try std.fmt.bufPrint(&dir_buf, "{s}/{s}/{s}/step{d:0>3}", .{ oracle, prompt, phase, step });
            var n: usize = 0;
            while (n < 512) : (n += 1) {
                _ = arena.reset(.retain_capacity);
                const y = (try read(u16, r, a, dir, "{d:0>3}_y", .{n})) orelse break;
                if (y.len == 0 or y.len % kdim != 0) return error.UnexpectedDump;
                const xs = (try read(f32, r, a, dir, "{d:0>3}_xs", .{n})) orelse return error.FileNotFound;
                try v.y(y, xs);
                files += 1;
                if (files % 8 == 0) try v.add(.{ .x = try present(try read(u16, r, a, dir, "{d:0>3}_h", .{n})) });
                if (n < 2 or n % 2 != 0) continue;
                norms += 1;
                if (norms % 4 != 0) continue;
                const x = try read(u16, r, a, dir, "{d:0>3}_h", .{n - 1});
                const res = try read(u16, r, a, dir, "m{d:0>2}_out", .{n / 2 - 1});
                const h = try read(u16, r, a, dir, "{d:0>3}_h", .{n});
                if (x == null or res == null or h == null or x.?.len != res.?.len or x.?.len != h.?.len) continue;
                fed += 1;
                equal += @intFromBool(for (x.?, res.?, h.?) |xe, re, he| {
                    if (ref.sumH(xe, re) != c.f(he)) break false;
                } else true);
                try v.add(.{ .x = x.?, .r = res.? });
            }
            if (n == 0 or std.mem.eql(u8, phase, "prefill")) break;
        }
    };
    std.debug.print("   {d} dumped y files, {d} residual calls ({d} with the dumped h equal to the host's rounded sum)\n", .{ files, fed, equal });
}
