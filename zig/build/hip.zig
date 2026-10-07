//! HIP build: one exact-architecture code object per -Dhip-arch entry, embedded together.
const std = @import("std");

/// Architectures with a qualified build; the runtime still loads only the device's exact match.
pub const supported = [_][]const u8{ "gfx1150", "gfx1151", "gfx1201" };

/// Code objects of `source` for each -Dhip-arch entry, in a directory whose `src` exports `archs` and `objects`.
pub const Objects = struct { files: *std.Build.Step.WriteFile, src: []const u8 };

pub fn objects(b: *std.Build, hipcc: []const u8, archs: []const u8, source: []const u8, name: []const u8) Objects {
    const files = b.addWriteFiles();
    var names: []const u8 = "";
    var decls: []const u8 = "";
    var count: usize = 0;
    var it = std.mem.tokenizeScalar(u8, archs, ',');
    while (it.next()) |arch| : (count += 1) {
        if (!isSupported(arch)) @panic(b.fmt("unsupported HIP architecture {s}", .{arch}));
        var seen = std.mem.tokenizeScalar(u8, archs, ',');
        for (0..count) |_| if (std.mem.eql(u8, seen.next().?, arch)) @panic(b.fmt("duplicate HIP architecture {s}", .{arch}));
        const compile = b.addSystemCommand(&.{ hipcc, "--genco", b.fmt("--offload-arch={s}", .{arch}), "-O2", "-ffp-contract=off" });
        compile.addFileArg(b.path(source));
        compile.addArg("-o");
        const object = compile.addOutputFileArg(b.fmt("{s}-{s}.hsaco", .{ name, arch }));
        _ = files.addCopyFile(object, b.fmt("{s}.hsaco", .{arch}));
        names = b.fmt("{s}\"{s}\", ", .{ names, arch });
        decls = b.fmt("{s}const o{d} align(8) = @embedFile(\"{s}.hsaco\").*;\n", .{ decls, count, arch });
    }
    if (count == 0) @panic("-Dhip-arch needs at least one architecture");
    var refs: []const u8 = "";
    for (0..count) |i| refs = b.fmt("{s}&o{d}, ", .{ refs, i });
    return .{ .files = files, .src = b.fmt("pub const archs = [_][]const u8{{ {s}}};\n{s}pub const objects = [_][]align(8) const u8{{ {s}}};\n", .{ names, decls, refs }) };
}

/// `hip_probe` module: the runtime-test kernels for every requested architecture.
pub fn probe(b: *std.Build, hipcc: []const u8, archs: []const u8) *std.Build.Module {
    const o = objects(b, hipcc, archs, "zig/kernels/hip/runtime_tests.hip", "hip-runtime-probe");
    return b.createModule(.{ .root_source_file = o.files.add("probe.zig", o.src) });
}

fn isSupported(arch: []const u8) bool {
    for (supported) |s| if (std.mem.eql(u8, s, arch)) return true;
    return false;
}
