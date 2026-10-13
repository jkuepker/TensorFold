//! The MTP head's draft vocabulary: the Python package's token ids the head scores (qwen4_exp/cuda/draft_vocab.txt).

const std = @import("std");

/// Whitespace-separated ids, each below `vocab`: sorted and unique, as np.unique gives them.
pub fn parse(gpa: std.mem.Allocator, text: []const u8, vocab: usize) ![]u32 {
    var ids: std.ArrayList(u32) = .empty;
    errdefer ids.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, " \n\r\t");
    while (it.next()) |tok| {
        const v = try std.fmt.parseInt(u32, tok, 10);
        if (v >= vocab) return error.DraftIdPastVocabulary;
        try ids.append(gpa, v);
    }
    if (ids.items.len == 0) return error.NoDraftIds;
    std.mem.sort(u32, ids.items, {}, std.sort.asc(u32));
    var n: usize = 1;
    for (ids.items[1..]) |v| {
        if (v == ids.items[n - 1]) continue;
        ids.items[n] = v;
        n += 1;
    }
    ids.shrinkRetainingCapacity(n);
    return ids.toOwnedSlice(gpa);
}

/// The draft vocabulary (draft_ids.txt beside this file), embedded at build time.
pub fn load(gpa: std.mem.Allocator, vocab: usize) ![]u32 {
    return parse(gpa, @embedFile("draft_ids.txt"), vocab);
}

test "draft ids sort, drop repeats and refuse ids past the vocabulary" {
    const a = std.testing.allocator;
    const ids = try parse(a, "5 3 3\n9 1", 10);
    defer a.free(ids);
    try std.testing.expectEqualSlices(u32, &.{ 1, 3, 5, 9 }, ids);
    try std.testing.expectError(error.DraftIdPastVocabulary, parse(a, "1 10", 10));
    try std.testing.expectError(error.NoDraftIds, parse(a, " ", 10));
}

test "the embedded list holds the Python package's 79,591 ids" {
    const a = std.testing.allocator;
    const ids = try load(a, 248320);
    defer a.free(ids);
    try std.testing.expectEqual(@as(usize, 79591), ids.len);
}
