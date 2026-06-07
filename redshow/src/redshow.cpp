#include <redshow.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "analysis/torch_view.h"  // new model
#include "analysis/torch_monitor.h"
#include "analysis/data_dependency.h"
#include "analysis/memory_liveness.h"
#include "analysis/memory_heatmap.h"
#include "analysis/memory_profile.h"
#include "analysis/data_flow.h"
#include "analysis/spatial_redundancy.h"
#include "analysis/temporal_redundancy.h"
#include "analysis/value_pattern.h"
#include "binutils/cubin.h"
#include "binutils/instruction.h"
#include "binutils/real_pc.h"
#include "binutils/symbol.h"
#include "common/map.h"
#include "common/set.h"
#include "common/utils.h"
#include "common/vector.h"
#include "operation/kernel.h"
#include "operation/memcpy.h"
#include "operation/memory.h"
#include "operation/memset.h"
#include "operation/memfree.h"

#include "torch_monitor.h"

#ifdef DEBUG
#define PRINT(...) fprintf(stderr, __VA_ARGS__)
#else
#define PRINT(...)
#endif

using namespace redshow;

/*
 * Global data structures
 */

static LockableMap<uint32_t, Cubin> cubin_map;

static LockableMap<uint32_t, CubinCache> cubin_cache_map;


// MemoryMap(<range, object>) is a map with current object
typedef Map<MemoryRange, std::shared_ptr<Memory>> MemoryMap;
// <op_id, memory_map>
static LockableMap<uint64_t, MemoryMap> memory_snapshot;

// reuse memory structure for sub-allocation
static LockableMap<uint64_t, MemoryMap> sub_memory_snapshot;

// Init analysis instance
// TODO(Keren): Separate address and full analysis modes
static Map<redshow_analysis_type_t, std::shared_ptr<Analysis>> analysis_enabled;

static Map<redshow_analysis_type_t, std::string> output_dir;

static redshow_log_data_callback_func log_data_callback = NULL;

static redshow_record_data_callback_func record_data_callback = NULL;

static redshow_get_op_id update_op_id_func = NULL;

static thread_local uint64_t mini_host_op_id = 0;

static uint32_t pc_views_limit = PC_VIEWS_LIMIT;
static uint32_t mem_views_limit = MEM_VIEWS_LIMIT;

static int decimal_degree_f32 = VALID_FLOAT_DIGITS;
static int decimal_degree_f64 = VALID_DOUBLE_DIGITS;

static redshow_data_type_t default_data_type = REDSHOW_DATA_UNKNOWN;

static std::mutex mtx;  // the lock for torch_view functional callback

unsigned long redshow_torchview_ongpu_get_range_size() {
  for (auto iter : analysis_enabled){
    if (iter.first == REDSHOW_ANALYSIS_TORCH_VIEW) {
      std::shared_ptr<redshow::TorchView> analysis_ptr = std::static_pointer_cast<redshow::TorchView>(iter.second);
      if (analysis_ptr->torchview_memory_snapshot.empty() && !analysis_ptr->_input_viewnode_forest_ptrs.empty()) {
        for (auto [idx, ptr] : analysis_ptr->_input_viewnode_forest_ptrs) {
          analysis_ptr->updata_torchview_memory_snapshot(ptr);
        }
      }
      return analysis_ptr->torchview_memory_snapshot.size();
    }
  }
  return 0;
}

void redshow_torchview_ongpu_set_ongpu() {
  for (auto iter : analysis_enabled){
    if (iter.first == REDSHOW_ANALYSIS_TORCH_VIEW) {
      std::shared_ptr<redshow::TorchView> analysis_ptr = std::static_pointer_cast<redshow::TorchView>(iter.second);
      analysis_ptr->torch_view_ongpu = true;
    }
  }
}

int redshow_torchview_ongpu_get_ongpu() {
  for (auto iter : analysis_enabled){
    if (iter.first == REDSHOW_ANALYSIS_TORCH_VIEW) {
      std::shared_ptr<redshow::TorchView> analysis_ptr = std::static_pointer_cast<redshow::TorchView>(iter.second);
      return (analysis_ptr->torch_view_ongpu)? 1 : 0;
    }
  }
  return 0;
}


void redshow_torchview_assemble_patch_analysis_address(gpu_patch_analysis_address_t* viewnode_ranges_host) {
  for (auto iter : analysis_enabled){
    if (iter.first == REDSHOW_ANALYSIS_TORCH_VIEW) {
      std::shared_ptr<redshow::TorchView> analysis_ptr = std::static_pointer_cast<redshow::TorchView>(iter.second);
      // size_t range_size = analysis_ptr->torchview_memory_snapshot.size();
      for (auto iter = analysis_ptr->torchview_memory_snapshot.begin(); iter != analysis_ptr->torchview_memory_snapshot.end(); iter++) {
        size_t offset = std::distance(analysis_ptr->torchview_memory_snapshot.begin(), iter);
        (viewnode_ranges_host + offset)->start = iter->first.start;
        (viewnode_ranges_host + offset)->end = iter->first.end;
      }
    }
  }
}


static void torch_memory_callback(torch_monitor_callback_site_t callback_site,
                                  torch_monitor_callback_data_t* callback_data) {
  if (callback_site == TORCH_MONITOR_CALLBACK_ENTER) {
#ifdef DEBUG
    std::cout << "Torch_Domain: " << callback_data->domain << std::endl;
#endif
      if (callback_data->domain == TORCH_MONITOR_DOMAIN_MEMORY
          && callback_data->data.mem_data.device_type == TORCH_MONITOR_DEVICE_TYPE_GPU) {
#ifdef DEBUG
        std::cout << "Current thread id: " << callback_data->current_thread_id << std::endl;
#endif
        if (callback_data->data.mem_data.type == TORCH_MONITOR_MEM_DATA_ALLOC) {
#ifdef DEBUG
          std::cout << "Allocate ptr: " << std::hex << callback_data->data.mem_data.ptr << std::dec
                    << std::endl;
#endif
          u64 op_id = update_op_id_func();
          redshow_result_t res = redshow_submemory_register(
            0, op_id, (u64) callback_data->data.mem_data.ptr,
            (u64) callback_data->data.mem_data.ptr + callback_data->data.mem_data.size
          );

        } else {
#ifdef DEBUG
          std::cout << "Free ptr: 0x" << std::hex << callback_data->data.mem_data.ptr << std::dec
                    << std::endl;
#endif
          u64 op_id = update_op_id_func();
          redshow_submemory_unregister(
            0, op_id, (u64) callback_data->data.mem_data.ptr,
            (u64) callback_data->data.mem_data.ptr + callback_data->data.mem_data.size
          );
      
        }
#ifdef DEBUG
      std::cout << "Size: " << callback_data->data.mem_data.size << std::endl;
      std::cout << "Total size: " << callback_data->data.mem_data.total_allocated << std::endl;
      std::cout << "Total reserved: " << callback_data->data.mem_data.total_reserved << std::endl;
#endif
    }
  }
}

/**
 * Handle with the PyTorch function callback inputs/outputs data storage.
 *
 * */
static void torch_callback_inputs_outputs_report(torch_monitor_callback_site_t callback_site,
                                           torch_monitor_callback_data_t* callback_data) {
  int64_t num_data = callback_data->data.op_data.input_output_data.size;
  if (callback_site == TORCH_MONITOR_CALLBACK_ENTER) {
    std::cout << "Callback inputs length: " << num_data << std::endl;
  } else if (callback_site == TORCH_MONITOR_CALLBACK_EXIT) {
    std::cout << "Callback outputs length: " << num_data << std::endl;
  }
  if (num_data <= 0)
    return;
  for (int64_t i = 0; i < num_data; i++) {
    torch_monitor_callback_tensor_data_t tensor =
          callback_data->data.op_data.input_output_data.tensor_data[i];
    if (tensor.index == -1 || tensor.numel <= 0)
      continue;
    std::cout << "  Index: " << tensor.index + 1 << std::endl;
    std::cout << "    Tensor elements: " << (std::int64_t)tensor.numel << std::endl;
    std::cout << "    Tensor dim: " << (std::int64_t)tensor.dim << std::endl;
    std::cout << "    Tensor dtype: " << torch_monitor_dtype_name_get(tensor.dtype)
              << std::endl;
    std::cout << "    Tensor itemsize: " << tensor.itemsize
              << std::endl;
    std::cout << "    Tensor storage offset: " << (std::int64_t)tensor.storage_offset
              << std::endl;
    std::cout << "    Tensor sizes: ";
    for (int64_t j = 0;
         j < std::min<int64_t>(TORCH_MONITOR_MAX_TENSOR_DIMENSION, tensor.dim); j++) {
      std::cout << tensor.sizes[j] << ", ";
    }
    std::cout << std::endl;
    std::cout << "    Tensor strides: ";
    for (int64_t j = 0;
         j < std::min<int64_t>(TORCH_MONITOR_MAX_TENSOR_DIMENSION, tensor.dim); j++) {
      std::cout << tensor.strides[j] << ", ";
    }
    std::cout << std::endl;
    std::cout << "    Tensor block address: " << /**std::hex <<*/ (u64)tensor.data_ptr << std::dec
              << std::endl;
    std::cout << "    Intrusive Ptr: " << /**std::hex <<*/ (u64)tensor.metadata_ptr << std::dec
              << std::endl;
  }
}

static void python_state_report() {
#ifdef DEBUG
  size_t num_states = 0;
  // Allow empty states
  torch_monitor_python_state_get(MAX_NUM_STATES, python_states, &num_states);
  for (size_t i = 0; i < num_states; ++i) {
    std::cout << "(" << i << ") "
              << "File: " << std::string(python_states[i].file_name) << std::endl;
    std::cout << "\tFunction: " << std::string(python_states[i].function_name)
              << std::endl;
    std::cout << "\tFirst line: " << python_states[i].function_first_lineno << std::endl;
    std::cout << "\tCall at line: " << python_states[i].lineno << std::endl;
  }
#endif
}

/**
 * for the model 'torch_view'
 * Handle with the PyTorch function callback info
 * */
static void torch_view_callback(torch_monitor_callback_site_t callback_site,
                                  torch_monitor_callback_data_t* callback_data) {
  std::lock_guard<std::mutex> lock_guard(mtx);
  // std::cout << "-----------------------------------" << std::endl;
  if (callback_site == TORCH_MONITOR_CALLBACK_ENTER) {
#ifdef DEBUG
    std::cout << "Enter Domain: " << callback_data->domain << std::endl;
#endif
    if (callback_data->domain != TORCH_MONITOR_DOMAIN_MEMORY) {
#ifdef DEBUG
      std::cout << "Current thread id: " << callback_data->current_thread_id << std::endl;
      std::cout << "Forward thread id: " << callback_data->data.op_data.forward_thread_id
                << std::endl;
      std::cout << "Sequence number: " << callback_data->data.op_data.sequence_number
                << std::endl;
      std::cout << "Name: " << std::string(callback_data->data.op_data.name) << std::endl;
#endif
      if (true) {  //timestamp_enable) {
#ifdef DEBUG
        std::cout << "Enter level: " << callback_data->data.op_data.nested_level << " at "
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::system_clock::now().time_since_epoch()).count()
                  << std::endl;
#endif
      }
      if (true) { //python_state_enable) {
        // python_state_report();
        // if (num_states != 0) {
        //   num__delayed_states = num_states;
        //   for(size_t i = 0; i < num_states; i++) {
        //     strcpy(delayed_python_states[i].file_name, python_states[i].file_name);
        //     delayed_python_states[i].function_first_lineno = python_states[i].function_first_lineno;
        //     strcpy(delayed_python_states[i].function_name, python_states[i].function_name);
        //     delayed_python_states[i].lineno = python_states[i].lineno;
        //   } // ready for delayed pystate insertion
        // }
        torch_monitor_python_state_get(MAX_NUM_STATES, python_states, &num_states);
        // if (num_states > 0) {
        //   torch_monitor_python_state_get(MAX_NUM_STATES, delayed_python_states, &num__delayed_states);
        // }
      }
      if (torch_monitor_inputs_capture_enable_get()) {
        /**
         * push the PyTorch functional callback op data into the op_stack
         */
        for (auto iter : analysis_enabled){
          if (iter.first == REDSHOW_ANALYSIS_TORCH_VIEW) {
            std::shared_ptr<redshow::TorchView> analysis_ptr = std::static_pointer_cast<redshow::TorchView>(iter.second);
            analysis_ptr->_op_stack.push(callback_data->data.op_data);
            analysis_ptr->_domain_name.push(std::string(callback_data->data.op_data.name));  // use together with PythonState
            // clear previous domain inputs tensor info(input tensors' respective ptr in view forest shadow memory)
            analysis_ptr->_input_viewnode_forest_ptrs.clear();
            analysis_ptr->torchview_memory_snapshot.clear();
            for (int64_t i = 0 ; i < callback_data->data.op_data.input_output_data.size; i++) {
              torch_monitor_callback_tensor_data_t titer = callback_data->data.op_data.input_output_data.tensor_data[i];
              if (titer.index == -1 || titer.numel <= 0){
                analysis_ptr->_input_viewnode_forest_ptrs[i] = nullptr;
                continue;
              }
              u64 view_id = update_op_id_func();
              redshow::TorchView::ViewNode * input_viewnode_forest_ptr = analysis_ptr->update_view_forest(titer, view_id, true);  // update the view forest
              analysis_ptr->_input_viewnode_forest_ptrs[i] = input_viewnode_forest_ptr;
              // std::cout << "redshow input: " << input_viewnode_forest_ptr->view_id << std::endl;
            }
          }
        }  // ends stack operation
#ifdef DEBUG
        torch_callback_inputs_outputs_report(callback_site, callback_data);
#endif
      }
    } else {  // callback_data->domain == TORCH_MONITOR_DOMAIN_MEMORY
#ifdef DEBUG
      std::cout << "Current thread id: " << callback_data->current_thread_id << std::endl;
#endif
      if (callback_data->data.mem_data.type == TORCH_MONITOR_MEM_DATA_ALLOC) {
#ifdef DEBUG
        std::cout << "Allocate ptr: " << std::hex << callback_data->data.mem_data.ptr
                  << std::dec << std::endl;
#endif
        u64 mem_block_id = update_op_id_func();
        /**
         * register a memory block
         * */
        for (auto iter : analysis_enabled){
          if (iter.first == REDSHOW_ANALYSIS_TORCH_VIEW) {
            std::shared_ptr<redshow::TorchView> analysis_ptr = std::static_pointer_cast<redshow::TorchView>(iter.second);
            analysis_ptr->register_memory_block(mem_block_id, callback_data->data.mem_data.device_type, callback_data->data.mem_data.ptr,
                                                callback_data->data.mem_data.size, callback_data->data.mem_data.total_allocated,
                                                callback_data->data.mem_data.total_reserved);
          }
        } // ends memory block registration
      } else {
#ifdef DEBUG
        std::cout << "Free ptr: " << std::hex << callback_data->data.mem_data.ptr
                  << std::dec << std::endl;
#endif
        /**
         * remove an entire tree from the view_node forest;
         **/
        for (auto aiter : analysis_enabled){
          if (aiter.first == REDSHOW_ANALYSIS_TORCH_VIEW) {
            std::shared_ptr<redshow::TorchView> analysis_ptr = std::static_pointer_cast<redshow::TorchView>(aiter.second);
            analysis_ptr->delete_forest_tree(output_dir[aiter.first],
                                             (redshow::TorchView::data_ptr_t)callback_data->data.mem_data.ptr,
                                             (int64_t)callback_data->data.mem_data.total_allocated);
//#ifdef DEBUG
//            for (auto titer : analysis_ptr->_roots){
//              analysis_ptr->visualize_view_forest(titer);
//            }
//#endif
          }
        }  // ends removing View Nodes
        /**
         * unregister a memory block
         * */
        for (auto iter : analysis_enabled){
          if (iter.first == REDSHOW_ANALYSIS_TORCH_VIEW) {
            std::shared_ptr<redshow::TorchView> analysis_ptr = std::static_pointer_cast<redshow::TorchView>(iter.second);
            analysis_ptr->unregister_memory_block(callback_data->data.mem_data.ptr);
          }
        } // ends memory block unregistration
      }
#ifdef DEBUG
      if (callback_data->data.mem_data.device_type == TORCH_MONITOR_DEVICE_TYPE_CPU) {
        std::cout << "Device: CPU" << std::endl;
      } else if (callback_data->data.mem_data.device_type = TORCH_MONITOR_DEVICE_TYPE_GPU) {
        std::cout << "Device: GPU" << std::endl;
      } else {
        std::cout << "Device: Other" << std::endl;
      }
        std::cout << "Size: " << callback_data->data.mem_data.size << std::endl;
        std::cout << "Total size: " << callback_data->data.mem_data.total_allocated
                  << std::endl;
        std::cout << "Total reserved: " << callback_data->data.mem_data.total_reserved
                  << std::endl;
#endif
    } // end callback_data->domain == TORCH_MONITOR_DOMAIN_MEMORY
  } else if (callback_site == TORCH_MONITOR_CALLBACK_EXIT) {
    if (callback_data->domain != TORCH_MONITOR_DOMAIN_MEMORY) {
      if (true) {  //timestamp_enable) {
#ifdef DEBUG
        std::cout << "Exit level: " << callback_data->data.op_data.nested_level << " at "
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(
                     std::chrono::system_clock::now().time_since_epoch()).count()
                  << std::endl;
#endif
      }
    }
#ifdef DEBUG
    std::cout << "Exit Domain: " << callback_data->domain << std::endl;
    std::cout << "Current thread id: " << callback_data->current_thread_id << std::endl;
    std::cout << "Forward thread id: " << callback_data->data.op_data.forward_thread_id
              << std::endl;
    std::cout << "Sequence number: " << callback_data->data.op_data.sequence_number
              << std::endl;
    std::cout << "Name: " << std::string(callback_data->data.op_data.name) << std::endl;
#endif
    if (torch_monitor_outputs_capture_enable_get()) {
      /**
       * pop the top element from the op_stack
       */
      for (auto iter : analysis_enabled){
        if (iter.first == REDSHOW_ANALYSIS_TORCH_VIEW) {
          std::shared_ptr<redshow::TorchView> analysis_ptr = std::static_pointer_cast<redshow::TorchView>(iter.second);
          if (analysis_ptr->_op_stack.size() > 0) {
            analysis_ptr->_popped_op = analysis_ptr->_op_stack.top();
            analysis_ptr->_popped_domain_name = analysis_ptr->_domain_name.top();
            for (int64_t i = 0 ; i < callback_data->data.op_data.input_output_data.size; i ++) {
              torch_monitor_callback_tensor_data_t titer = callback_data->data.op_data.input_output_data.tensor_data[i];
              if (titer.index == -1 || titer.numel <= 0)
                continue;
              u64 view_id = update_op_id_func();
              analysis_ptr->update_view_forest(titer, view_id, false);  // update the view forest but no need to update torchiew-ongpu(if enabled) torchview_memory_snapshot
            }
            analysis_ptr->torchview_memory_snapshot.clear();
            analysis_ptr->_op_stack.pop();
            analysis_ptr->_domain_name.pop();
          }
#ifdef DEBUG
          // for (auto titer : analysis_ptr->_roots){
          //   analysis_ptr->visualize_view_forest(titer);
          // }
#endif
          // clear delayed unit access trace
          analysis_ptr->map_delayed_access();
          analysis_ptr->clear_delayed_trace();
        }
      }
#ifdef DEBUG
      torch_callback_inputs_outputs_report(callback_site, callback_data);
#endif
    }
  } //callback_site == TORCH_MONITOR_CALLBACK_EXIT
}

static redshow_result_t analyze_cubin(const char *path, SymbolVector &symbols,
                                      InstructionGraph &inst_graph) {
  redshow_result_t result = REDSHOW_SUCCESS;

  std::string cubin_path = std::string(path);
  auto iter = cubin_path.rfind("/");
  if (iter == std::string::npos) {
    result = REDSHOW_ERROR_NO_SUCH_FILE;
  } else {
    // x/x.cubin
    // 012345678
    std::string cubin_name = cubin_path.substr(iter + 1, cubin_path.size() - iter);
    // x/cubins/x.cubin
    iter = cubin_path.rfind("/", iter - 1);
    std::string dir_name = cubin_path.substr(0, iter);
    std::string inst_path = dir_name + "/structs/nvidia/" + cubin_name + ".inst";

    // Prevent boost from core dump
    std::ifstream f(inst_path.c_str());
    if (f.good() == false) {
      result = REDSHOW_ERROR_NO_SUCH_FILE;
    } else {
      // instructions are analyzed before hpcrun
      if (InstructionParser::parse(inst_path, symbols, inst_graph)) {
        result = REDSHOW_SUCCESS;
      } else {
        result = REDSHOW_ERROR_FAILED_ANALYZE_CUBIN;
      }
    }
  }

  return result;
}

static redshow_result_t trace_analyze_address_patch(int32_t kernel_id, u64 host_op_id, InstructionGraph *inst_graph,
                                                    SymbolVector *symbols, MemoryMap *memory_map,
                                                    gpu_patch_buffer_t *trace_data) {
  redshow_result_t result = REDSHOW_SUCCESS;

  size_t size = trace_data->head_index;
  gpu_patch_record_address_t *records =
      reinterpret_cast<gpu_patch_record_address_t *>(trace_data->records);

  // Dummy entries
  AccessKind access_kind;
  ThreadId thread_id{0, 0};

  for (size_t i = 0; i < size; ++i) {
    // Iterate over each record
    gpu_patch_record_address_t *record = records + i;

    for (size_t j = 0; j < GPU_PATCH_WARP_SIZE; ++j) {
      if ((record->active & (0x1u << j)) == 0) {
        continue;
      }

      // <start, end>
      MemoryRange memory_range(record->address[j], record->address[j] + record->size);

      auto iter = memory_map->prev(memory_range);
      uint64_t memory_op_id = 0;
      int32_t memory_id = 0;
      uint64_t memory_size = 0;
      uint64_t memory_addr = 0;
      if (iter != memory_map->end()) {
        if (record->address[j] >= iter->second->memory_range.start &&
            record->address[j] + record->size <= iter->second->memory_range.end) {
          memory_op_id = iter->second->op_id;
          memory_id = iter->second->ctx_id;
          memory_size = record->size;
          memory_addr = memory_range.start;
        } else {
          // TODO(Keren): Investigate what are the causes
          // Prevent out of bound memory accesses
          continue;
        }
      }

      if (memory_op_id == 0) {
        // Unknown memory object
        continue;
      }

      // only for memory profile heatmap storage compression
      access_kind.unit_size = record->size;

      // start get pc
      RealPC real_pc;

      auto ret = symbols->transform_pc(record->pc);
      if (ret.has_value()) {
        real_pc = ret.value();
        // std::cout << "Symbol.index: " << real_pc.function_index << std::endl;
      } else {
        result = REDSHOW_ERROR_FAILED_ANALYZE_CUBIN;
        return result;
      }
      // end get pc

      Memory memory = Memory(memory_op_id, memory_id, memory_addr, memory_size);
      // XXX(Keren): Need to separate address analysis with value analysis
      for (auto aiter : analysis_enabled) {
        aiter.second->unit_access(kernel_id, host_op_id, thread_id, access_kind, memory, real_pc.pc_offset, 0, 0, 0,
                                  static_cast<GPUPatchFlags>(record->flags));
      }
    }
  }
  return result;
}

static redshow_result_t trace_analyze_address_analysis(int32_t kernel_id, u64 host_op_id, MemoryMap *memory_map,
                                                       gpu_patch_buffer_t *trace_data) {
  redshow_result_t result = REDSHOW_SUCCESS;

  size_t size = trace_data->head_index;
  gpu_patch_analysis_address_t *records =
      reinterpret_cast<gpu_patch_analysis_address_t *>(trace_data->records);

  // Dummy entries
  AccessKind access_kind;
  ThreadId thread_id{0, 0};

  for (size_t i = 0; i < size; ++i) {
    // Iterate over each record
    gpu_patch_analysis_address_t *record = records + i;

    // <start, end>
    MemoryRange memory_range(record->start, record->end);
    uint64_t addr_start = record->start;
    uint64_t addr_end = addr_start;

    // Separate memories from a continous address region
    while (addr_end < memory_range.end) {
      MemoryRange cur_memory_range(addr_start, addr_start);
      auto iter = memory_map->prev(cur_memory_range);

      uint64_t memory_op_id = 0;
      int32_t memory_id = 0;
      uint64_t memory_size = 0;
      uint64_t memory_addr = 0;
      if (iter != memory_map->end()) {
        if (iter->first.start <= addr_start && iter->first.end > addr_start) {
          addr_end = MIN2(memory_range.end, iter->first.end);
          memory_op_id = iter->second->op_id;
          memory_id = iter->second->ctx_id;
          memory_size = addr_end - addr_start;
          memory_addr = addr_start;
          addr_start = iter->first.end;
        } else {
          // TODO(Keren): Investigate what are the causes
          // Prevent out of bound memory accesses
          break;
        }
      }

      if (memory_op_id == 0) {
        // Unknown memory object
        break;
      }

      Memory memory = Memory(memory_op_id, memory_id, memory_addr, memory_size);
      // XXX(Keren): Need to separate address analysis with value analysis
      for (auto aiter : analysis_enabled) {
        aiter.second->unit_access(kernel_id, host_op_id, thread_id, access_kind, memory, 0, 0, 0, 0,
                                  static_cast<GPUPatchFlags>(trace_data->flags));
      }
    }
  }

  return result;
}

static redshow_result_t trace_analyze_default(int32_t kernel_id, u64 host_op_id, InstructionGraph *inst_graph,
                                              SymbolVector *symbols, MemoryMap *memory_map,
                                              gpu_patch_buffer_t *trace_data) {
  redshow_result_t result = REDSHOW_SUCCESS;

  size_t size = trace_data->head_index;
  gpu_patch_record_t *records = reinterpret_cast<gpu_patch_record_t *>(trace_data->records);

  for (size_t i = 0; i < size; ++i) {
    // Iterate over each record
    gpu_patch_record_t *record = records + i;

    if (record->size == 0) {
      // Fast path, no thread active
      continue;
    }

    if (record->flags & GPU_PATCH_BLOCK_ENTER_FLAG) {
      // Skip analysis
    } else if (record->flags & GPU_PATCH_BLOCK_EXIT_FLAG) {
      // Remove temporal records
      for (size_t j = 0; j < GPU_PATCH_WARP_SIZE; ++j) {
        if (record->active & (0x1u << j)) {
          uint32_t flat_thread_id =
              record->flat_thread_id / GPU_PATCH_WARP_SIZE * GPU_PATCH_WARP_SIZE + j;
          ThreadId thread_id{record->flat_block_id, flat_thread_id};
          for (auto aiter : analysis_enabled) {
            aiter.second->block_exit(thread_id);
          }
        }
      }
    } else {
      RealPC real_pc;

      auto ret = symbols->transform_pc(record->pc);
      if (ret.has_value()) {
        real_pc = ret.value();
      } else {
        result = REDSHOW_ERROR_FAILED_ANALYZE_CUBIN;
        return result;
      }

      // record->size * 8, byte to bits
      AccessKind access_kind;

      if (inst_graph->size() != 0) {
        // Accurate mode, when we have instruction information
        auto &inst = inst_graph->node(real_pc.cubin_offset);
        if (inst.access_kind.get() != NULL) {
          access_kind = *inst.access_kind;
        }
        // Fall back to default mode if failed
      }

      if (access_kind.data_type == REDSHOW_DATA_UNKNOWN) {
        // Default mode, we identify every data as 64 bits unit size, 64 bits vec size, float type
        access_kind.data_type = default_data_type;
        if (access_kind.vec_size == 0) {
          access_kind.vec_size = record->size * 8;
          access_kind.unit_size = MIN2(32, access_kind.vec_size);
        }
      }

      // Reserved for debugging
      // std::cout << "function_index: " << real_pc.function_index << ", pc_offset: " <<
      //  real_pc.pc_offset << ", " << access_kind.to_string() << std::endl;
      // TODO: accelerate by handling all threads in a warp together
      for (size_t j = 0; j < GPU_PATCH_WARP_SIZE; ++j) {
        if ((record->active & (0x1u << j)) == 0) {
          continue;
        }

        uint32_t flat_thread_id =
            record->flat_thread_id / GPU_PATCH_WARP_SIZE * GPU_PATCH_WARP_SIZE + j;
        ThreadId thread_id{record->flat_block_id, flat_thread_id};

        MemoryRange memory_range(record->address[j], record->address[j]);
        auto iter = memory_map->prev(memory_range);
        uint64_t memory_op_id = 0;
        int32_t memory_id = 0;
        uint64_t memory_size = 0;
        uint64_t memory_addr = 0;
        if (iter != memory_map->end()) {
          if (record->address[j] >= iter->second->memory_range.start &&
              record->address[j] + record->size <= iter->second->memory_range.end) {
            memory_op_id = iter->second->op_id;
            memory_id = iter->second->ctx_id;
            memory_size = iter->second->len;
            memory_addr = iter->second->memory_range.start;
          } else {
            // TODO(Keren): Investigate what are the causes
            // Prevent out of bound memory accesses
            continue;
          }
        }

        uint32_t stride = GLOBAL_MEMORY_OFFSET;
        if (memory_op_id == 0) {
          // XXX(Keren): memory_op_id == 1 ?
          // Memory object not found, it means the memory is local, shared, or allocated in an
          // unknown way
          if (record->flags & GPU_PATCH_LOCAL) {
            memory_op_id = REDSHOW_MEMORY_LOCAL;
            memory_id = LOCAL_MEMORY_CTX_ID;
            stride = LOCAL_MEMORY_OFFSET;
          } else if (record->flags & GPU_PATCH_SHARED) {
            memory_op_id = REDSHOW_MEMORY_SHARED;
            memory_id = SHARED_MEMORY_CTX_ID;
            stride = SHARED_MEMORY_OFFSET;
          } else {
            // Unknown allocation
          }
        }

        if (memory_op_id == 0) {
          // Unknown memory object
          continue;
        }

        Memory memory = Memory(memory_op_id, memory_id, memory_addr, memory_size);
        auto num_units = access_kind.vec_size / access_kind.unit_size;
        AccessKind unit_access_kind = access_kind;
        // We iterate through all the units such that every unit's vec_size = unit_size
        unit_access_kind.vec_size = unit_access_kind.unit_size;

        for (size_t m = 0; m < num_units; m++) {
          uint64_t value = 0;
          uint32_t byte_size = unit_access_kind.unit_size >> 3u;
          memcpy(&value, &record->value[j][m * byte_size], byte_size);
          // The 7th digit in float number's decimal part is partial valid, so we set the deault
          // approx level to REDSHOW_APPROX_MIN.
          value =
              unit_access_kind.value_to_basic_type(value, decimal_degree_f32, decimal_degree_f64);

          // Reserved for debug
          // std::cout << "thread: " << j << ", value: " << value << std::endl;

          for (auto aiter : analysis_enabled) {
            aiter.second->unit_access(kernel_id, host_op_id, thread_id, unit_access_kind, memory, record->pc,
                                      value, record->address[j], m,
                                      static_cast<GPUPatchFlags>(record->flags));
          }
        }
      }
    }
  }

  return result;
}

static redshow_result_t trace_analyze(uint32_t cpu_thread, uint32_t cubin_id, uint32_t mod_id,
                                      int32_t kernel_id, uint64_t host_op_id, uint32_t stream_id,
                                      gpu_patch_buffer_t *trace_data) {
  redshow_result_t result = REDSHOW_SUCCESS;

  SymbolVector *symbols = NULL;
  InstructionGraph *inst_graph = NULL;
  // Cubin path is added just for debugging purpose
  std::string cubin_path;

  cubin_map.lock();
  if (!cubin_map.has(cubin_id) || !cubin_map.at(cubin_id).symbols.has(mod_id)) {
    result = REDSHOW_ERROR_NOT_EXIST_ENTRY;
  } else {
    symbols = &(cubin_map.at(cubin_id).symbols.at(mod_id));
    inst_graph = &(cubin_map.at(cubin_id).inst_graph);
    cubin_path = cubin_map.at(cubin_id).path;
  }
  cubin_map.unlock();

  // Cubin not found, maybe in the cache map
  if (result == REDSHOW_ERROR_NOT_EXIST_ENTRY) {
    uint32_t nsymbols;
    uint64_t *symbol_pcs;
    const char *path;

    cubin_cache_map.lock();
    if (!cubin_cache_map.has(cubin_id)) {
      result = REDSHOW_ERROR_NOT_EXIST_ENTRY;
    } else {
      auto &cubin_cache = cubin_cache_map.at(cubin_id);
      if (!cubin_cache.symbol_pcs.has(mod_id)) {
        result = REDSHOW_ERROR_NOT_EXIST_ENTRY;
      } else {
        result = REDSHOW_SUCCESS;
        nsymbols = cubin_cache.nsymbols;
        symbol_pcs = cubin_cache.symbol_pcs.at(mod_id).get();
        path = cubin_cache.path.c_str();
      }
    }
    cubin_cache_map.unlock();

    if (result == REDSHOW_SUCCESS) {
      result = redshow_cubin_register(cubin_id, mod_id, nsymbols, symbol_pcs, path);
    }

    // Try fetch cubin again
    if (result == REDSHOW_SUCCESS) {
      cubin_map.lock();
      if (!cubin_map.has(cubin_id)) {
        result = REDSHOW_ERROR_NOT_EXIST_ENTRY;
      } else {
        auto &cubin = cubin_map.at(cubin_id);
        if (!cubin.symbols.has(mod_id)) {
          result = REDSHOW_ERROR_NOT_EXIST_ENTRY;
        } else {
          result = REDSHOW_SUCCESS;
          symbols = &(cubin.symbols.at(mod_id));
          inst_graph = &(cubin.inst_graph);
          cubin_path = cubin.path;
        }
      }
      cubin_map.unlock();
    }
  }

  if (result != REDSHOW_SUCCESS) {
    return result;
  }

  MemoryMap *memory_map = NULL;

  memory_snapshot.lock();
  auto snapshot_iter = memory_snapshot.prev(host_op_id);
  if (snapshot_iter == memory_snapshot.end()) {
    result = REDSHOW_ERROR_NOT_EXIST_ENTRY;
  } else {
    memory_map = &(snapshot_iter->second);
  }
  memory_snapshot.unlock();

  // Memory snapshot not found
  if (result != REDSHOW_SUCCESS) {
    return result;
  }

  for (auto aiter : analysis_enabled) {
    aiter.second->analysis_begin(cpu_thread, kernel_id, host_op_id, stream_id, cubin_id, mod_id,
                                 static_cast<GPUPatchType>(trace_data->type), trace_data);
  }

  if (trace_data->type == GPU_PATCH_TYPE_DEFAULT) {
    result = trace_analyze_default(kernel_id, host_op_id, inst_graph, symbols, memory_map, trace_data);
  } else if (trace_data->type == GPU_PATCH_TYPE_ADDRESS_PATCH) {
    result = trace_analyze_address_patch(kernel_id, host_op_id, inst_graph, symbols, memory_map, trace_data);
  } else if (trace_data->type == GPU_PATCH_TYPE_ADDRESS_ANALYSIS) {
    result = trace_analyze_address_analysis(kernel_id, host_op_id, memory_map, trace_data);
  }

  for (auto aiter : analysis_enabled) {
    aiter.second->analysis_end(cpu_thread, kernel_id);
  }

  return result;
}

/*
 * Interface methods
 */

redshow_result_t redshow_output_dir_config(redshow_analysis_type_t analysis, const char *dir) {
  PRINT("\nredshow-> Enter redshow_output_dir_config\nanalysis: %u\ndir: %s\n", analysis, dir);

  if (dir) {
    output_dir[analysis] = std::string(dir);
  }

  return REDSHOW_SUCCESS;
}

redshow_result_t redshow_data_type_config(redshow_data_type_t data_type) {
  PRINT("\nredshow-> Enter redshow_data_type_config\ndata_type: %u\n", data_type);

  redshow_result_t result = REDSHOW_SUCCESS;

  switch (data_type) {
    case REDSHOW_DATA_FLOAT:
      default_data_type = REDSHOW_DATA_FLOAT;
      break;
    case REDSHOW_DATA_INT:
      default_data_type = REDSHOW_DATA_INT;
      break;
    default:
      result = REDSHOW_ERROR_NO_SUCH_DATA_TYPE;
      break;
  }

  return result;
};

redshow_result_t redshow_data_type_get(redshow_data_type_t *data_type) {
  *data_type = default_data_type;

  return REDSHOW_SUCCESS;
}

redshow_result_t redshow_approx_level_config(redshow_approx_level_t level) {
  redshow_result_t result = REDSHOW_SUCCESS;

  switch (level) {
    case REDSHOW_APPROX_NONE:
      decimal_degree_f32 = VALID_FLOAT_DIGITS;
      decimal_degree_f64 = VALID_DOUBLE_DIGITS;
      break;
    case REDSHOW_APPROX_MIN:
      decimal_degree_f32 = MIN_FLOAT_DIGITS;
      decimal_degree_f64 = MIN_DOUBLE_DIGITS;
      break;
    case REDSHOW_APPROX_LOW:
      decimal_degree_f32 = LOW_FLOAT_DIGITS;
      decimal_degree_f64 = LOW_DOUBLE_DIGITS;
      break;
    case REDSHOW_APPROX_MID:
      decimal_degree_f32 = MID_FLOAT_DIGITS;
      decimal_degree_f64 = MID_DOUBLE_DIGITS;
      break;
    case REDSHOW_APPROX_HIGH:
      decimal_degree_f32 = HIGH_FLOAT_DIGITS;
      decimal_degree_f64 = HIGH_DOUBLE_DIGITS;
      break;
    case REDSHOW_APPROX_MAX:
      decimal_degree_f32 = MAX_FLOAT_DIGITS;
      decimal_degree_f64 = MAX_DOUBLE_DIGITS;
      break;
    default:
      result = REDSHOW_ERROR_NO_SUCH_APPROX;
      break;
  }

  return result;
}

redshow_result_t redshow_approx_get(int *degree_f32, int *degree_f64) {
  redshow_result_t result = REDSHOW_SUCCESS;

  *degree_f32 = decimal_degree_f32;
  *degree_f64 = decimal_degree_f64;

  return result;
}

redshow_result_t redshow_torch_enable() {
  torch_monitor_status status = torch_monitor_domain_enable(TORCH_MONITOR_DOMAIN_FUNCTION);
  if (status != TORCH_MONITOR_STATUS_SUCCESS) {
    std::cerr << "Torch monitor status: " << status << std::endl;
    exit(1);
  } 

  status = torch_monitor_domain_enable(TORCH_MONITOR_DOMAIN_BACKWARD_FUNCTION);
  if (status != TORCH_MONITOR_STATUS_SUCCESS) {
    std::cerr << "Torch monitor status: " << status << std::endl;
    exit(1);
  } 
  
  status = torch_monitor_domain_enable(TORCH_MONITOR_DOMAIN_MEMORY);
  if (status != TORCH_MONITOR_STATUS_SUCCESS) {
    std::cerr << "Torch monitor status: " << status << std::endl;
    exit(1);
  }    
  status = torch_monitor_callback_subscribe(torch_memory_callback);
  if (status != TORCH_MONITOR_STATUS_SUCCESS) {
    std::cerr << "Torch monitor status: " << status << std::endl;
    exit(1);
  }
  status = torch_monitor_init();
  if (status != TORCH_MONITOR_STATUS_SUCCESS) {
    std::cerr << "Torch monitor status: " << status << std::endl;
    exit(1);
  }

  return REDSHOW_SUCCESS;   
}

redshow_result_t redshow_torch_view_enable() {
  /**
   * enable inputs outputs capturing
   */
  torch_monitor_inputs_capture_enable_set(true);
  torch_monitor_outputs_capture_enable_set(true);
  if (!(torch_monitor_inputs_capture_enable_get() && torch_monitor_outputs_capture_enable_get())) {
    std::cerr << "Inputs and Outputs capturing is not enabled." << std::endl;
    exit(1);
  }

  torch_monitor_status status = torch_monitor_domain_enable(TORCH_MONITOR_DOMAIN_FUNCTION);
  if (status != TORCH_MONITOR_STATUS_SUCCESS) {
    std::cerr << "Torch monitor status: " << status << std::endl;
    exit(1);
  }

  status = torch_monitor_domain_enable(TORCH_MONITOR_DOMAIN_BACKWARD_FUNCTION);
  if (status != TORCH_MONITOR_STATUS_SUCCESS) {
    std::cerr << "Torch monitor status: " << status << std::endl;
    exit(1);
  }

  status = torch_monitor_domain_enable(TORCH_MONITOR_DOMAIN_MEMORY);
  if (status != TORCH_MONITOR_STATUS_SUCCESS) {
    std::cerr << "Torch monitor status: " << status << std::endl;
    exit(1);
  }
  /**
   * "torch_view_callback"
   * the updated driver func that enabled function callback inputs/outputs capturing
   * */
  status = torch_monitor_callback_subscribe(torch_view_callback);
  if (status != TORCH_MONITOR_STATUS_SUCCESS) {
    std::cerr << "Torch monitor status: " << status << std::endl;
    exit(1);
  }
  status = torch_monitor_init();
  if (status != TORCH_MONITOR_STATUS_SUCCESS) {
    std::cerr << "Torch monitor status: " << status << std::endl;
    exit(1);
  }

  return REDSHOW_SUCCESS;
}

// invoked by sanitizer-api.c
redshow_result_t redshow_get_op_id_register(redshow_get_op_id func) {
  update_op_id_func = func;

  return REDSHOW_SUCCESS;
}

/**
 *
 * Create Analysis Object
 * @param analysis_type
 * @return
 */
redshow_result_t redshow_analysis_enable(redshow_analysis_type_t analysis_type) {
  PRINT("\nredshow-> Enter redshow_analysis_enable\nanalysis_type: %u\n", analysis_type);
  printf("Enter Sanitizer %u enable \n", analysis_type);
  redshow_result_t result = REDSHOW_SUCCESS;

  switch (analysis_type) {
    case REDSHOW_ANALYSIS_SPATIAL_REDUNDANCY:
      analysis_enabled.emplace(REDSHOW_ANALYSIS_SPATIAL_REDUNDANCY,
                               std::make_shared<SpatialRedundancy>());
      break;
    case REDSHOW_ANALYSIS_TEMPORAL_REDUNDANCY:
      analysis_enabled.emplace(REDSHOW_ANALYSIS_TEMPORAL_REDUNDANCY,
                               std::make_shared<TemporalRedundancy>());
      break;
    case REDSHOW_ANALYSIS_DATA_FLOW:
      analysis_enabled.emplace(REDSHOW_ANALYSIS_DATA_FLOW, std::make_shared<DataFlow>());
      break;
    case REDSHOW_ANALYSIS_VALUE_PATTERN:
      analysis_enabled.emplace(REDSHOW_ANALYSIS_VALUE_PATTERN, std::make_shared<ValuePattern>());
      break;
    case REDSHOW_ANALYSIS_MEMORY_PROFILE:
      analysis_enabled.emplace(REDSHOW_ANALYSIS_MEMORY_PROFILE, std::make_shared<MemoryProfile>());
      break;
    case REDSHOW_ANALYSIS_MEMORY_HEATMAP:
      analysis_enabled.emplace(REDSHOW_ANALYSIS_MEMORY_HEATMAP, std::make_shared<MemoryHeatmap>());
      break;
    case REDSHOW_ANALYSIS_MEMORY_LIVENESS:
      analysis_enabled.emplace(REDSHOW_ANALYSIS_MEMORY_LIVENESS, std::make_shared<MemoryLiveness>());
      break;
    case REDSHOW_ANALYSIS_DATA_DEPENDENCY:
      analysis_enabled.emplace(REDSHOW_ANALYSIS_DATA_DEPENDENCY, std::make_shared<DataDependency>());
      break;
    case REDSHOW_ANALYSIS_TORCH_MONITOR:
      analysis_enabled.emplace(REDSHOW_ANALYSIS_TORCH_MONITOR, std::make_shared<TorchMonitor>());
      break;
    case REDSHOW_ANALYSIS_TORCH_VIEW:
      analysis_enabled.emplace(REDSHOW_ANALYSIS_TORCH_VIEW, std::make_shared<TorchView>());
      break;
    default:
      result = REDSHOW_ERROR_NO_SUCH_ANALYSIS;
      break;
  }

  return result;
}

redshow_result_t redshow_analysis_enabled(redshow_analysis_type_t analysis_type) {
  PRINT("\nredshow-> Enter redshow_analysis_enabled\nanalysis_type: %u\n", analysis_type);

  redshow_result_t result = REDSHOW_SUCCESS;

  if (!analysis_enabled.has(analysis_type)) {
    result = REDSHOW_ERROR_NOT_EXIST_ENTRY;
  }

  return result;
}

redshow_result_t redshow_analysis_disable(redshow_analysis_type_t analysis_type) {
  PRINT("\nredshow-> Enter redshow_analysis_disable\nanalysis_type: %u\n", analysis_type);

  analysis_enabled.erase(analysis_type);

  return REDSHOW_SUCCESS;
}

redshow_result_t redshow_analysis_config(redshow_analysis_type_t analysis_type,
                                         redshow_analysis_config_type_t config_type, bool enable) {
  PRINT(
      "\nredshow-> Enter redshow_analysis_config\nanalysis_type: %u, config_type: %u, enable: %u\n",
      analysis_type, config_type, enable);

  if (analysis_enabled.has(analysis_type)) {
    analysis_enabled[analysis_type]->config(config_type, enable);
  }

  return REDSHOW_SUCCESS;
}

redshow_result_t redshow_cubin_register(uint32_t cubin_id, uint32_t mod_id, uint32_t nsymbols,
                                        const uint64_t *symbol_pcs, const char *path) {
  PRINT("\nredshow-> Enter redshow_cubin_register\ncubin_id: %u\nmode_id: %u\npath: %s\n", cubin_id,
        mod_id, path);

  redshow_result_t result = REDSHOW_SUCCESS;

  InstructionGraph inst_graph;
  SymbolVector symbols(nsymbols);
  result = analyze_cubin(path, symbols, inst_graph);

  if (result == REDSHOW_SUCCESS || result == REDSHOW_ERROR_NO_SUCH_FILE) {
    // We must have found an instruction file, no matter nvdisasm failed or not
    // Assign symbol pc
    for (auto i = 0; i < nsymbols; ++i) {
      symbols[i].pc = symbol_pcs[i];
    }

    // Sort symbols by pc
    std::sort(symbols.begin(), symbols.end());

    cubin_map.lock();

    if (!cubin_map.has(cubin_id)) {
      cubin_map[cubin_id].cubin_id = cubin_id;
      cubin_map[cubin_id].path = path;
      cubin_map[cubin_id].inst_graph = inst_graph;
      result = REDSHOW_SUCCESS;
    } else if (cubin_map[cubin_id].symbols.find(mod_id) == cubin_map[cubin_id].symbols.end()) {
      result = REDSHOW_SUCCESS;
    } else {
      result = REDSHOW_ERROR_DUPLICATE_ENTRY;
    }
    if (result != REDSHOW_ERROR_DUPLICATE_ENTRY) {
      cubin_map[cubin_id].symbols[mod_id] = symbols;
    }

    cubin_map.unlock();
  }

  return result;
}

redshow_result_t redshow_cubin_cache_register(uint32_t cubin_id, uint32_t mod_id, uint32_t nsymbols,
                                              uint64_t *symbol_pcs, const char *path) {
  PRINT("\nredshow-> Enter redshow_cubin_cache_register\ncubin_id: %u\nmod_id: %u\npath: %s\n",
        cubin_id, mod_id, path);

  redshow_result_t result = REDSHOW_SUCCESS;

  cubin_cache_map.lock();
  if (!cubin_cache_map.has(cubin_id)) {
    auto &cubin_cache = cubin_cache_map[cubin_id];

    cubin_cache.cubin_id = cubin_id;
    cubin_cache.path = std::string(path);
    cubin_cache.nsymbols = nsymbols;
    result = REDSHOW_SUCCESS;
  } else if (cubin_cache_map[cubin_id].symbol_pcs.find(mod_id) ==
             cubin_cache_map[cubin_id].symbol_pcs.end()) {
    result = REDSHOW_SUCCESS;
  } else {
    result = REDSHOW_ERROR_DUPLICATE_ENTRY;
  }

  if (result != REDSHOW_ERROR_DUPLICATE_ENTRY) {
    auto *pcs = new uint64_t[nsymbols];
    cubin_cache_map[cubin_id].symbol_pcs[mod_id].reset(pcs);
    for (size_t i = 0; i < nsymbols; ++i) {
      pcs[i] = symbol_pcs[i];
    }
  }
  cubin_cache_map.unlock();

  return result;
}

redshow_result_t redshow_cubin_unregister(uint32_t cubin_id, uint32_t mod_id) {
  PRINT("\nredshow-> Enter redshow_cubin_unregister\ncubin_id: %u\n", cubin_id);

  redshow_result_t result = REDSHOW_SUCCESS;

  cubin_map.lock();
  if (cubin_map.has(cubin_id)) {
    cubin_map.at(cubin_id).symbols.erase(mod_id);
    if (cubin_map.at(cubin_id).symbols.size() == 0) {
      cubin_map.erase(cubin_id);
    }
    result = REDSHOW_SUCCESS;
  } else {
    result = REDSHOW_ERROR_NOT_EXIST_ENTRY;
  }
  cubin_map.unlock();

  return result;
}

redshow_result_t redshow_memory_register(uint32_t stream_id, int32_t memory_id, uint64_t host_op_id,
                                         uint64_t start, uint64_t end) {
  PRINT(
      "\nredshow-> Enter redshow_memory_register\nstream_id: %u\nmemory_id: %d\nhost_op_id: %lu\n"
      "start: %lu\nend: %lu\n",
      stream_id, memory_id, host_op_id, start, end);

  redshow_result_t result = REDSHOW_SUCCESS;

  MemoryMap memory_map;
  MemoryRange memory_range(start, end);
  auto memory = std::make_shared<Memory>(stream_id, host_op_id, memory_id, memory_range);

  memory_snapshot.lock();
  if (memory_snapshot.size() == 0) {
    // First snapshot
    memory_map[memory_range] = memory;
    memory_snapshot[host_op_id] = memory_map;
    result = REDSHOW_SUCCESS;
    PRINT("Register memory_id %d\n", memory_id);
  } else {
    // snapshot_iter's op_id <= host_op_id
    auto iter = memory_snapshot.prev(host_op_id);
    if (iter != memory_snapshot.end()) {
      // Take a snapshot
      memory_map = iter->second;
      if (!memory_map.has(memory_range)) {
        memory_map[memory_range] = memory;
        memory_snapshot[host_op_id] = memory_map;
        result = REDSHOW_SUCCESS;
        PRINT("Register memory_id %d\n", memory_id);
      } else {
        result = REDSHOW_ERROR_DUPLICATE_ENTRY;
      }
    } else {
      result = REDSHOW_ERROR_NOT_EXIST_ENTRY;
    }
  }
  memory_snapshot.unlock();

  if (result == REDSHOW_SUCCESS) {
    for (auto aiter : analysis_enabled) {
      aiter.second->op_callback(memory);
    }
  }

  return result;
}

redshow_result_t redshow_memory_unregister(uint32_t stream_id, int32_t memory_id, uint64_t host_op_id,
                                           uint64_t start, uint64_t end) {
  PRINT("\nredshow-> Enter redshow_memory_unregister\nstream_id: %u\nmemory_free_id: %d\n"
        "host_op_id: %lu\nstart: %lu\nend: %lu\n",
        stream_id, memory_id, host_op_id, start, end);

  redshow_result_t result = REDSHOW_SUCCESS;

  MemoryMap memory_map;
  MemoryRange memory_range(start, end);
  auto memfree = std::make_shared<Memfree>(stream_id, host_op_id, memory_id, memory_range);

  memory_snapshot.lock();
  auto snapshot_iter = memory_snapshot.prev(host_op_id);
  if (snapshot_iter != memory_snapshot.end()) {
    // Take a snapshot
    memory_map = snapshot_iter->second;
    auto memory_map_iter = memory_map.find(memory_range);
    if (memory_map_iter != memory_map.end()) {
      memory_map.erase(memory_map_iter);
      memory_snapshot[host_op_id] = memory_map;
      result = REDSHOW_SUCCESS;
    } else {
      result = REDSHOW_ERROR_NOT_EXIST_ENTRY;
    }
  } else {
    result = REDSHOW_ERROR_NOT_EXIST_ENTRY;
  }
  memory_snapshot.unlock();

  if (result == REDSHOW_SUCCESS) {
    for (auto aiter : analysis_enabled) {
      aiter.second->op_callback(memfree);
    }
  }

  return result;
}

redshow_result_t redshow_submemory_register(int32_t sub_memory_id, uint64_t host_op_id,
                                                 uint64_t start, uint64_t end) {
  PRINT("\nredshow-> Enter redshow_submemory_register\nmemory_id: %d\nhost_op_id: %lu\nstart: %lu\nend: %lu\n",
  sub_memory_id, host_op_id, start, end);

  redshow_result_t result = REDSHOW_SUCCESS;

  MemoryMap sub_memory_map;
  MemoryRange sub_memory_range(start, end);

  auto submemory = std::make_shared<Memory>(0, host_op_id, sub_memory_id, sub_memory_range);

  sub_memory_snapshot.lock();
    if (sub_memory_snapshot.size() == 0) {
        // First sub-memory snapshot
        sub_memory_map[sub_memory_range] = submemory;
        sub_memory_snapshot[host_op_id] = sub_memory_map;
        result = REDSHOW_SUCCESS;
        PRINT("Register sub-memory_id %d\n", sub_memory_id);
    } else {
        auto iter = sub_memory_snapshot.prev(host_op_id);
        if (iter != sub_memory_snapshot.end()) {
            // Take a snapshot
            sub_memory_map = iter->second;
            if (!sub_memory_map.has(sub_memory_range)) {
                sub_memory_map[sub_memory_range] = submemory;
                sub_memory_snapshot[host_op_id] = sub_memory_map;
                result = REDSHOW_SUCCESS;
                PRINT("Register memory_id %d\n", sub_memory_id);
            } else {
                result = REDSHOW_ERROR_DUPLICATE_ENTRY;
            }
        } else {
            result = REDSHOW_ERROR_NOT_EXIST_ENTRY;
        }
    }
    sub_memory_snapshot.unlock();

    bool is_submemory = true;
    if (result == REDSHOW_SUCCESS) {
        for (auto aiter : analysis_enabled) {

            aiter.second->op_callback(submemory, is_submemory);
        }
    }

    return result;    

}

redshow_result_t redshow_submemory_unregister(int32_t sub_memory_id, uint64_t host_op_id, uint64_t start, 
                                           uint64_t end) {
  PRINT("\nredshow-> Enter redshow_submemory_unregister\nmemory_free_id: %d\nhost_op_id: %lu\nstart: %lu\nend: %lu\n",
        sub_memory_id, host_op_id, start, end);

  redshow_result_t result = REDSHOW_SUCCESS;

  MemoryMap sub_memory_map;
  MemoryRange sub_memory_range(start, end);
  auto submemfree = std::make_shared<Memfree>(0, host_op_id, sub_memory_id, sub_memory_range);

  sub_memory_snapshot.lock();
  auto snapshot_iter = sub_memory_snapshot.prev(host_op_id);
  if (snapshot_iter != sub_memory_snapshot.end()) {
    // Take a snapshot
    sub_memory_map = snapshot_iter->second;
    auto sub_memory_map_iter = sub_memory_map.find(sub_memory_range);
    if (sub_memory_map_iter != sub_memory_map.end()) {
      sub_memory_map.erase(sub_memory_map_iter);
      sub_memory_snapshot[host_op_id] = sub_memory_map;
      result = REDSHOW_SUCCESS;
    } else {
      result = REDSHOW_ERROR_NOT_EXIST_ENTRY;
    }
  } else {
    result = REDSHOW_ERROR_NOT_EXIST_ENTRY;
  }
  sub_memory_snapshot.unlock();

  bool is_submemory = true;
  if (result == REDSHOW_SUCCESS) {
    for (auto aiter : analysis_enabled) {
      aiter.second->op_callback(submemfree, is_submemory);
    }
  }

  return result;
}

redshow_result_t redshow_memory_query(uint64_t host_op_id, uint64_t start, int32_t *memory_id,
                                      uint64_t *memory_op_id, uint64_t *shadow_start,
                                      uint64_t *len) {
  PRINT("\nredshow-> Enter redshow_memory_query\nhost_op_id: %lu\nstart: %lu\n", host_op_id, start);

  redshow_result_t result = REDSHOW_SUCCESS;

  MemoryRange memory_range(start, 0);

  memory_snapshot.lock();
  auto snapshot_iter = memory_snapshot.prev(host_op_id);
  if (snapshot_iter != memory_snapshot.end()) {
    auto &memory_map = snapshot_iter->second;
    auto memory_map_iter = memory_map.prev(memory_range);
    if (memory_map_iter != memory_map.end() && memory_map_iter->first.end >= start) {
      // Fall into the range, assume no memory overflow
      *memory_id = memory_map_iter->second->ctx_id;
      *memory_op_id = memory_map_iter->second->op_id;
      auto offset = start - memory_map_iter->first.start;
      *shadow_start = reinterpret_cast<uint64_t>(memory_map_iter->second->value.get()) + offset;
      *len = memory_map_iter->second->len;
      PRINT("memory_id: %d\nmemory_op_id: %lu\noffset %lu\nshadow: %lu\nlen: %lu\n", *memory_id,
            *memory_op_id, offset, *shadow_start, *len);
      result = REDSHOW_SUCCESS;
    } else {
      result = REDSHOW_ERROR_NOT_EXIST_ENTRY;
    }
  } else {
    result = REDSHOW_ERROR_NOT_EXIST_ENTRY;
  }
  memory_snapshot.unlock();

  return result;
}

EXTERNC redshow_result_t redshow_memory_ranges_get(uint64_t host_op_id, uint64_t limit,
                                                   gpu_patch_analysis_address *start_end,
                                                   uint32_t *len) {
  PRINT("\nredshow-> Enter redshow_memory_ranges_get\nhost_op_id: %lu\nlimit: %lu\n", host_op_id,
        limit);

  redshow_result_t result = REDSHOW_SUCCESS;

  *len = 0;
  memory_snapshot.lock();
  auto snapshot_iter = memory_snapshot.prev(host_op_id);
  if (snapshot_iter != memory_snapshot.end()) {
    auto &memory_map = snapshot_iter->second;
    if (memory_map.size() != 0) {
      auto memory_map_iter = memory_map.begin();
      while (memory_map_iter != memory_map.end()) {
        start_end[(*len)].start = memory_map_iter->first.start;
        start_end[(*len)].end = memory_map_iter->first.end;
        ++(*len);
        ++memory_map_iter;
      }
    } else {
      result = REDSHOW_ERROR_NOT_EXIST_ENTRY;
    }
  } else {
    result = REDSHOW_ERROR_NOT_EXIST_ENTRY;
  }
  memory_snapshot.unlock();

  return result;
}


EXTERNC redshow_result_t redshow_submemory_ranges_get(uint64_t host_op_id, uint64_t limit,
                                                   gpu_patch_analysis_address *start_end,
                                                   uint32_t *len) {
  PRINT("\nredshow-> Enter redshow_submemory_ranges_get\nhost_op_id: %lu\nlimit: %lu\n", host_op_id,
        limit);

  redshow_result_t result = REDSHOW_SUCCESS;

  *len = 0;
  sub_memory_snapshot.lock();
  auto snapshot_iter = sub_memory_snapshot.prev(host_op_id);
  if (snapshot_iter != sub_memory_snapshot.end()) {
    auto &memory_map = snapshot_iter->second;
    if (memory_map.size() != 0) {
      auto memory_map_iter = memory_map.begin();
      while (memory_map_iter != memory_map.end()) {
        start_end[(*len)].start = memory_map_iter->first.start;
        start_end[(*len)].end = memory_map_iter->first.end;
        ++(*len);
        ++memory_map_iter;
      }
    } else {
      result = REDSHOW_ERROR_NOT_EXIST_ENTRY;
    }
  } else {
    result = REDSHOW_ERROR_NOT_EXIST_ENTRY;
  }
  sub_memory_snapshot.unlock();

  return result;
}

redshow_result_t redshow_memcpy_register(int32_t memcpy_id, uint64_t host_op_id, bool src_host, uint32_t src_stream_id,
                                         uint64_t src_start, bool dst_host, uint32_t dst_stream_id, uint64_t dst_start,
                                         uint64_t len) {
  PRINT(
      "\nredshow-> Enter redshow_memcpy_register\nmemcpy_id: %d\nhost_op_id: "
      "%lu\nsrc_host: %d\nsrc_stream_id: %u\nsrc_start: %lu\ndst_host: %d\ndst_stream_id: %u\ndst_start: "
      "%lu\nlen: %lu\n",
      memcpy_id, host_op_id, src_host, src_stream_id, src_start, dst_host, dst_stream_id, dst_start, len);

  redshow_result_t result = REDSHOW_SUCCESS;

  i32 src_mem_id = 0;
  u64 src_mem_op_id = 0;
  u64 src_mem_addr = 0;
  u64 src_size = 0;
  i32 dst_mem_id = 0;
  u64 dst_mem_op_id = 0;
  u64 dst_mem_addr = 0;
  u64 dst_size = 0;

  if (src_host) {
    src_mem_addr = src_start;
    src_mem_id = REDSHOW_MEMORY_HOST;
    src_mem_op_id = REDSHOW_MEMORY_HOST;
  } else {
    redshow_memory_query(host_op_id, src_start, &src_mem_id, &src_mem_op_id, &src_mem_addr,
                         &src_size);
  }

  if (dst_host) {
    dst_mem_addr = dst_start;
    dst_mem_id = REDSHOW_MEMORY_HOST;
    dst_mem_op_id = REDSHOW_MEMORY_HOST;
  } else {
    redshow_memory_query(host_op_id, dst_start, &dst_mem_id, &dst_mem_op_id, &dst_mem_addr,
                         &dst_size);
  }

  if (dst_mem_addr != 0 && src_mem_addr != 0) {
    // Avoid memcpy to symbol without allocation
    auto memcpy = std::make_shared<Memcpy>(host_op_id, memcpy_id, src_mem_op_id, src_stream_id, src_start,
                                           src_mem_addr, dst_mem_op_id, dst_stream_id, dst_start,
                                           dst_mem_addr, len);

    for (auto aiter : analysis_enabled) {
      aiter.second->op_callback(memcpy);
    }
  }

  return result;
}

redshow_result_t redshow_memset_register(uint32_t stream_id, int32_t memset_id, uint64_t host_op_id, uint64_t start,
                                         uint32_t value, uint64_t len) {
  PRINT(
      "\nredshow-> Enter redshow_memset_register\nstream_id%u\nmemset_id: %d\nhost_op_id: %lu\nstart: "
      "%lu\nvalue: %u\nlen: %lu\n",
      stream_id, memset_id, host_op_id, start, value, len);

  redshow_result_t result = REDSHOW_SUCCESS;

  i32 mem_id = 0;
  u64 mem_op_id = 0;
  u64 addr = 0;
  u64 size = 0;

  redshow_memory_query(host_op_id, start, &mem_id, &mem_op_id, &addr, &size);

  auto memset = std::make_shared<Memset>(host_op_id, memset_id, mem_op_id, stream_id, start, addr, value, len);

  if (addr != 0) {
    for (auto aiter : analysis_enabled) {
      aiter.second->op_callback(memset);
    }
  }

  return result;
}

redshow_result_t redshow_log_data_callback_register(redshow_log_data_callback_func func) {
  log_data_callback = func;

  return REDSHOW_SUCCESS;
}

redshow_result_t redshow_record_data_callback_register(redshow_record_data_callback_func func,
                                                       uint32_t pc_views, uint32_t mem_views) {
  record_data_callback = func;
  pc_views_limit = pc_views;
  mem_views_limit = mem_views;

  return REDSHOW_SUCCESS;
}

redshow_result_t redshow_tool_dtoh_register(redshow_tool_dtoh_func func) {
  // std::cout << "modules we have: ";
  // std::cout << analysis_enabled.size() << std::endl;
  for (auto &aiter : analysis_enabled) {
    aiter.second->dtoh_register(func);
  }

  return REDSHOW_SUCCESS;
}

redshow_result_t redshow_pc_views_get(uint32_t *views) {
  *views = pc_views_limit;

  return REDSHOW_SUCCESS;
}

redshow_result_t redshow_mem_views_get(uint32_t *views) {
  *views = mem_views_limit;

  return REDSHOW_SUCCESS;
}

redshow_result_t redshow_kernel_begin(uint32_t cpu_thread, int32_t kernel_id, uint64_t host_op_id) {
  return REDSHOW_SUCCESS;
}

redshow_result_t redshow_kernel_end(uint32_t cpu_thread, uint32_t stream_id, int32_t kernel_id, uint64_t host_op_id) {
  PRINT("\nredshow-> Enter redshow_kernel_end\ncpu_thread: %u\nstream_id: %u\nkernel_id: %d\nhost_op_id: %lu\n",
        cpu_thread, stream_id, kernel_id, host_op_id);

  // propose changes
  // read_memory_op_ids, write_memory_op_ids
  redshow_result_t result = REDSHOW_SUCCESS;

  auto kernel = std::make_shared<Kernel>(host_op_id, kernel_id, cpu_thread, stream_id);

  for (auto aiter : analysis_enabled) {
    aiter.second->op_callback(kernel);
  }

  return REDSHOW_SUCCESS;
}

redshow_result_t redshow_analyze(uint32_t cpu_thread, uint32_t cubin_id, uint32_t mod_id,
                                 int32_t kernel_id, uint64_t host_op_id, uint32_t stream_id,
                                 gpu_patch_buffer_t *trace_data) {
  PRINT(
      "\nredshow-> Enter redshow_analyze\ncpu_thread: %u\ncubin_id: %u\nmod_id: %u\n"
      "kernel_id: %d\nhost_op_id: %lu\ntrace_data: %p\n",
      cpu_thread, cubin_id, mod_id, kernel_id, host_op_id, trace_data);

  redshow_result_t result;

  // Analyze trace_data
  result = trace_analyze(cpu_thread, cubin_id, mod_id, kernel_id, host_op_id, stream_id, trace_data);

  if (result == REDSHOW_SUCCESS) {
    if (log_data_callback) {
      log_data_callback(kernel_id, trace_data);
      if (mini_host_op_id == 0) {
        mini_host_op_id = host_op_id;
      } else {
        mini_host_op_id = MIN2(mini_host_op_id, host_op_id);
      }
      result = REDSHOW_SUCCESS;
    } else {
      result = REDSHOW_ERROR_NOT_REGISTER_CALLBACK;
    }
  } else {
    PRINT("\nredshow-> Fail redshow_analyze result %d\n", result);
  }

  return result;
}

redshow_result_t redshow_analysis_begin() {
  PRINT("\nredshow-> Enter redshow_analysis_begin\n");

  mini_host_op_id = 0;

  return REDSHOW_SUCCESS;
}

redshow_result_t redshow_analysis_end() {
  PRINT("\nredshow-> Enter redshow_analysis_end\n");

  redshow_result_t result;

  if (mini_host_op_id != 0 && !analysis_enabled.has(REDSHOW_ANALYSIS_DATA_FLOW)) {
    // Remove all the memory snapshots before mini_host_op_id
    Vector<uint64_t> ids;

    memory_snapshot.lock();
    uint64_t max_min_host_op_id = 0;
    for (auto &iter : memory_snapshot) {
      if (iter.first < mini_host_op_id) {
        ids.push_back(iter.first);
        max_min_host_op_id = MAX2(iter.first, max_min_host_op_id);
      }
    }
    // Maintain the largest snapshot
    for (auto &id : ids) {
      if (id == max_min_host_op_id) {
        continue;
      }
      memory_snapshot.erase(id);
    }
    memory_snapshot.unlock();

    result = REDSHOW_SUCCESS;
  } else {
    result = REDSHOW_ERROR_NOT_REGISTER_CALLBACK;
  }

  return result;
}

redshow_result_t redshow_flush_thread(uint32_t cpu_thread) {
  PRINT("\nredshow-> Enter redshow_flush cpu_thread %u\n", cpu_thread);

  for (auto aiter : analysis_enabled) {
    aiter.second->flush_thread(cpu_thread, output_dir[aiter.first], cubin_map,
                               record_data_callback);
  }

  return REDSHOW_SUCCESS;
}

redshow_result_t redshow_flush() {
  PRINT("\nredshow-> Enter redshow_flush\n");

  for (auto aiter : analysis_enabled) {
    aiter.second->flush(output_dir[aiter.first], cubin_map, record_data_callback);
  }

  return REDSHOW_SUCCESS;
}
