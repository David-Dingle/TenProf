#ifndef HPCTOOLKIT_GPU_PATCH_GPU_PATCH_H
#define HPCTOOLKIT_GPU_PATCH_GPU_PATCH_H

#include <stdbool.h>
#include <stdint.h>

#define GPU_PATCH_MAX_ACCESS_SIZE (16)
#define GPU_PATCH_WARP_SIZE (32)
#define GPU_PATCH_ANALYSIS_THREADS (1024)
#define GPU_PATCH_ANALYSIS_ITEMS (4)
#define GPU_PATCH_ADDRESS_DICT_SIZE (1024)
#define GPU_PATCH_DEBUGGING_CHAR_SIZE (1024) // Only for debugging

enum GPUPatchFlags {
  GPU_PATCH_NONE = 0,
  GPU_PATCH_READ = 0x1,
  GPU_PATCH_WRITE = 0x2,
  GPU_PATCH_ATOMSYS = 0x4,
  GPU_PATCH_LOCAL = 0x8,
  GPU_PATCH_SHARED = 0x10,
  GPU_PATCH_BLOCK_ENTER_FLAG = 0x20,
  GPU_PATCH_BLOCK_EXIT_FLAG = 0x40,
  GPU_PATCH_ANALYSIS = 0x80
};

enum GPUPatchType {
  GPU_PATCH_TYPE_DEFAULT = 0,
  GPU_PATCH_TYPE_ADDRESS_PATCH = 1,
  GPU_PATCH_TYPE_ADDRESS_ANALYSIS = 2,
  GPU_PATCH_TYPE_COUNT = 3
};

// Complete record
typedef struct gpu_patch_record {
  uint64_t pc;
  uint32_t size;
  uint32_t active;
  uint32_t flat_thread_id;
  uint32_t flat_block_id;
  uint32_t flags;
  uint64_t address[GPU_PATCH_WARP_SIZE];
  uint8_t value[GPU_PATCH_WARP_SIZE][GPU_PATCH_MAX_ACCESS_SIZE];  // STS.128->16 bytes
} gpu_patch_record_t;

// Address only
typedef struct gpu_patch_record_address {
  uint64_t pc;
  uint32_t flags;
  uint32_t active;
  uint32_t size;
  uint64_t address[GPU_PATCH_WARP_SIZE];
} gpu_patch_record_address_t;

// Address only, gpu analysis
typedef struct gpu_patch_analysis_address {
  uint64_t start;
  uint64_t end;
} gpu_patch_analysis_address_t;

// Auxiliary data
typedef struct gpu_patch_aux_address_dict {
  uint32_t size;
  gpu_patch_analysis_address_t start_end[GPU_PATCH_ADDRESS_DICT_SIZE];
  uint8_t hit[GPU_PATCH_ADDRESS_DICT_SIZE];
  uint8_t read[GPU_PATCH_ADDRESS_DICT_SIZE];
  uint8_t write[GPU_PATCH_ADDRESS_DICT_SIZE];
} gpu_patch_aux_address_dict_t;

/**
 * Auxiliary data for torch-view on-GPU func input tensors memory range hit
*/
typedef struct gpu_patch_aux_torchview_dict {
  uint64_t function_pc_offset; // Offset of current function starting pc. 
  uint32_t view_range_size; // computed by the host after PyTorch function tensor-typed inputs are known
  gpu_patch_analysis_address_t* start_end; // assign view_range_size * sizeof(gpu_patch_analysis_address_t) memory; Must BE Initialized with sorted order
  uint64_t current_read_pc_size; // (func-insts / 8)
  uint64_t current_write_pc_size; // (func-insts / 8) keep this redundancy incase we want to use precise sized map
  uint64_t* read_pc_range_bit_map; // a logical 2-D array with dim: [max_pc_size, ceiling(view_range_size / 64) + 1]; Initialize with Zeros
  uint64_t* write_pc_range_bit_map; // same
} gpu_patch_aux_torchview_dict_t;

typedef struct gpu_patch_buffer {
  volatile uint32_t full;
  volatile uint32_t analysis;
  volatile uint32_t head_index;
  volatile uint32_t tail_index;
  uint32_t size;
  uint32_t num_threads;  // If num_threads == 0, the kernel is finished
  uint32_t block_sampling_offset;
  uint32_t block_sampling_frequency;
  uint32_t type;
  uint32_t flags;  // read or write or both
  void *records;
  void *aux;  // useed by liveness and torchview
  void *torch_aux; // use with liveness + torch memory block traceing
} gpu_patch_buffer_t;

#endif
