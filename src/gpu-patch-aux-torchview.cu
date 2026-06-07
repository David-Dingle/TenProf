#include "gpu-patch.h"
#include "gpu-queue.h"
#include "utils.h"

#include <sanitizer_patching.h>

struct gpu_patch_analysis_address_comparator {
  __device__
  bool operator()(gpu_patch_analysis_address &l, gpu_patch_analysis_address &r) {
    return l.start <= r.start;
  }
};

/*
 * Monitor each shared and global memory access.
 */
static 
__device__ __forceinline__
SanitizerPatchResult
memory_access_callback
(
 void *user_data,
 uint64_t pc,
 void *address,
 uint32_t size,
 uint32_t flags,
 const void *new_value
) 
{
  gpu_patch_buffer_t *buffer = (gpu_patch_buffer_t *)user_data;

  // 1. Init values
  uint32_t active_mask = __activemask();

  uint32_t byte_offset = 0;
  if (buffer->aux != NULL && (flags & (GPU_PATCH_SHARED | GPU_PATCH_LOCAL)) == 0) {
    // Read address can be filtered
    gpu_patch_aux_torchview_dict_t *address_dict = (gpu_patch_aux_torchview_dict_t *)buffer->aux;
    gpu_patch_analysis_address_t *start_end = address_dict->start_end;
    gpu_patch_analysis_address_t addr = { (uint64_t)address, 0 };
    uint32_t pos = map_prev(start_end, addr, address_dict->view_range_size, gpu_patch_analysis_address_comparator());

    if (pos != address_dict->view_range_size && (uint64_t)address < (start_end + pos)->end) { // >= start_end[pos].end >= address >= start_end[pos].start
      byte_offset = (pos / 64); // column index at corresponding (pc) row
      uint64_t bit_mask = 1 << (pos % 64); // bit offset within above byte
      uint32_t range_columns = ((address_dict->view_range_size) / 64);
      // Step 1 Find/Insert pc from/into read/write_pc_range_bit_map; get the row index
      if (static_cast<GPUPatchFlags>(flags) == GPU_PATCH_READ) {
        uint64_t* read_pc_range_map = address_dict->read_pc_range_bit_map;
        const uint64_t local_pc_offset = (pc - address_dict->function_pc_offset) / 8;
        *(read_pc_range_map + (local_pc_offset * (range_columns + 2)) + byte_offset) |= bit_mask;
      } else if (static_cast<GPUPatchFlags>(flags) == GPU_PATCH_WRITE) {
        uint64_t* write_pc_range_map = address_dict->write_pc_range_bit_map;
        const uint64_t local_pc_offset = (pc - address_dict->function_pc_offset) / 8;
        *(write_pc_range_map + (local_pc_offset * (range_columns + 2)) + byte_offset) |= bit_mask;
      } 
    } 
  }
  __syncwarp(active_mask);
  return SANITIZER_PATCH_SUCCESS;
}


extern "C"
__device__ __noinline__
SanitizerPatchResult
sanitizer_global_memory_access_callback
(
 void *user_data,
 uint64_t pc,
 void *address,
 uint32_t size,
 uint32_t flags,
 const void *new_value
) 
{
  return memory_access_callback(user_data, pc, address, size, flags, new_value);
}


extern "C"
__device__ __noinline__
SanitizerPatchResult
sanitizer_shared_memory_access_callback
(
 void *user_data,
 uint64_t pc,
 void *address,
 uint32_t size,
 uint32_t flags,
 const void *new_value
) 
{
  return memory_access_callback(user_data, pc, address, size, flags | GPU_PATCH_SHARED, new_value);
}


extern "C"
__device__ __noinline__
SanitizerPatchResult
sanitizer_local_memory_access_callback
(
 void *user_data,
 uint64_t pc,
 void *address,
 uint32_t size,
 uint32_t flags,
 const void *new_value
) 
{
  return memory_access_callback(user_data, pc, address, size, flags | GPU_PATCH_LOCAL, new_value);
}


/*
 * Lock the corresponding hash entry for a block
 */
extern "C"
__device__ __noinline__
SanitizerPatchResult
sanitizer_block_exit_callback
(
 void *user_data,
 uint64_t pc
)
{
  gpu_patch_buffer_t* buffer = (gpu_patch_buffer_t *)user_data;

  if (!sample_callback(buffer->block_sampling_frequency, buffer->block_sampling_offset)) {
    return SANITIZER_PATCH_SUCCESS;
  }

  uint32_t active_mask = __activemask();
  uint32_t laneid = get_laneid();
  uint32_t first_laneid = __ffs(active_mask) - 1;
  int32_t pop_count = __popc(active_mask);

  if (laneid == first_laneid) {
    // Finish a bunch of threads
    atomicAdd(&buffer->num_threads, -pop_count);
  }

  return SANITIZER_PATCH_SUCCESS;
}