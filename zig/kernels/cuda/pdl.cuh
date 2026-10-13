// Programmatic dependent launch: a kernel launched with the PDL attribute may start its blocks while the grid before it is still running.
#pragma once

// First statement of every kernel launched with PDL: holds until the grid before it has finished and flushed its writes.
__device__ __forceinline__ void tf_grid_wait() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

// Lets the next PDL launch begin once every block of this grid has started (it still waits for this grid to finish).
__device__ __forceinline__ void tf_grid_launch() {
#if __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
