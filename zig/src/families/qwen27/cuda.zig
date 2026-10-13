//! Qwen3.8-27B on CUDA in Zig: the Python 0.6.6 engine's kernels and layouts, beside the family's Metal engine.

pub const config = @import("cuda/shape.zig");
pub const kernels = @import("cuda/kernels.zig");
pub const triton = @import("cuda/triton.zig");
pub const glue_native = @import("cuda/glue_native.zig");
pub const prep_native = @import("cuda/prep_native.zig");
pub const weights = @import("cuda/weights.zig");
pub const state = @import("cuda/state.zig");
pub const round = @import("cuda/round.zig");
pub const tree = @import("cuda/tree.zig");
pub const Forward = @import("cuda/forward.zig").Forward;
pub const Part = @import("cuda/forward.zig").Part;
pub const engine = @import("cuda/engine.zig");
pub const Engine = engine.Engine;
pub const Lanes = @import("cuda/lanes.zig").Cuda;
pub const native = @import("cuda/native.zig");
pub const reuse = @import("cuda/reuse.zig");
pub const draft = @import("cuda/draft.zig");
pub const lone = @import("cuda/lone.zig");
pub const drafts = @import("cuda/drafts.zig");
pub const features = @import("cuda/features.zig");

test {
    _ = config;
    _ = kernels;
    _ = triton;
    _ = glue_native;
    _ = weights;
    _ = state;
    _ = engine;
    _ = @import("cuda/forward.zig");
    _ = @import("cuda/lanes.zig");
    _ = reuse;
    _ = @import("cuda/tree.zig");
    _ = @import("cuda/draft_load.zig");
    _ = @import("cuda/draft_policy.zig");
    _ = draft;
    _ = lone;
    _ = native;
}
