//
// Created by xjding on 1/1/24.
//

#include "analysis/torch_view.h"

#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <set>
#include <string.h>

// TENPROF: the torch_view callback mutex, defined in redshow.cpp. The kernel-
// analysis methods below lock on THIS (instead of Analysis::_lock) so they
// serialize with torch_view_callback (the PyTorch-op callback), which uses the
// same mutex. Lock it AFTER any GIL-acquiring call to keep the GIL->mtx order.
std::mutex& redshow_torchview_mtx();

#include "torch_monitor.h"

#include "common/utils.h"

static FILE * fp;

namespace redshow {

  void TorchView::update_op_node(u64 op_id, i32 ctx_id) {}

  void TorchView::op_callback(OperationPtr op, bool is_submemory /* default = false */) {
        // Add a calling context node
    // TENPROF: was lock()/unlock() == Analysis::_lock, a DIFFERENT mutex than the
    // PyTorch-op callback's redshow_torchview_mtx() -> the two contexts raced on the
    // shared view-forest maps -> heap corruption on multi-threaded apps. Each
    // specific callback below now locks the SHARED redshow_torchview_mtx() itself
    // (kernel locks AFTER its GIL fetch, to preserve GIL->mtx order). Old wrapper:
    // lock();
    if (op->type == OPERATION_TYPE_KERNEL) {
      kernel_op_callback(std::dynamic_pointer_cast<Kernel>(op));
    } else if (op->type == OPERATION_TYPE_MEMORY) {
      memory_op_callback(std::dynamic_pointer_cast<Memory>(op), is_submemory);
    } else if (op->type == OPERATION_TYPE_MEMFREE) {
      memfree_op_callback(std::dynamic_pointer_cast<Memfree>(op), is_submemory);
    } else if (op->type == OPERATION_TYPE_MEMCPY) {
      memcpy_op_callback(std::dynamic_pointer_cast<Memcpy>(op));
    } else if (op->type == OPERATION_TYPE_MEMSET) {
      memset_op_callback(std::dynamic_pointer_cast<Memset>(op));
    }
    // unlock();
  }

  /**
    * Pass
    * In theory, we don't care about CUDA memory allocation
    * May empty this code block later
    */
  void TorchView::memory_op_callback(std::shared_ptr<Memory> op, bool is_submemory) {}

  /**
    * Pass
    * We don't care about CUDA level memory free operations
    * May empty this code block later
    */
  void TorchView::memfree_op_callback(std::shared_ptr<Memfree> op, bool is_submemory) {}

/**
 * Need fix
 * */
  void TorchView::kernel_op_callback(std::shared_ptr<Kernel> op) {
    // TENPROF deadlock fix (option 1): use the CACHED (no-GIL) python state instead
    // of a fresh fetch. This runs while the sanitizer holds the per-context
    // entry->lock; the fresh fetch acquires the GIL there and deadlocks (AB-BA)
    // against a thread holding the GIL and spinning on entry->lock. The cached
    // thread-local state is the launching op's stack (captured fresh, GIL-safe, at
    // the PyTorch op-enter on this same thread) -- the correct state for this kernel.
    // torch_monitor_python_state_get(MAX_NUM_STATES, delayed_python_states, &num__delayed_states);
    torch_monitor_python_state_get_cached(MAX_NUM_STATES, delayed_python_states, &num__delayed_states);
    // TENPROF race fix: hold the SHARED torch_view mutex across the whole analysis
    // (which mutates call_path_map / _roots / _input_viewnode_forest_ptrs /
    // torchview_memory_snapshot / ...). Taken AFTER the GIL-acquiring call above so
    // the order is GIL->mtx (matches torch_view_callback) -> no mtx<->GIL deadlock.
    // The analysis body below acquires no GIL, so holding mtx across it is safe.
    std::lock_guard<std::mutex> _tv_lk(redshow_torchview_mtx());
    std::string all_states("");
    if (num__delayed_states > 0 && num__delayed_states <= MAX_NUM_STATES) {
      for(size_t i = 0; i < num__delayed_states; i++) {
        all_states.append(delayed_python_states[i].file_name);
        all_states.append(delayed_python_states[i].function_name);
        all_states.append(std::to_string(delayed_python_states[i].function_first_lineno));
        all_states.append(std::to_string(delayed_python_states[i].lineno));
      }
    }
    aten_copy_pystate_hash = (std::size_t)std::hash<std::string>{}(all_states);


    if (!torch_view_ongpu) {
      // std::cout << "Enter TORCH_VIEW Kernel op callback." << std::endl;
      if (_trace.get() == NULL) {
        // If the kernel is sampled
        return;
      }

      std::map<uint64_t, std::vector<ViewNode*>> _pc_node_cache;

      // STEP 1
      // Initialize the the _delay_trace table
      // _delayed_trace = NULL; // reset delayed trace to NULL
      if (!_delayed_trace) {
        _delayed_trace = std::make_shared<TorchViewDelayedTrace>();
        // torch_monitor_python_state_get(MAX_NUM_STATES, delayed_python_states, &num__delayed_states);
        PyStateCTX _state{-1, num__delayed_states, delayed_python_states};
        if (num__delayed_states > 0) {
          if (!_domain_name.empty()) {
            strcat(_state.py_state[0].function_name, "^");
            strcat(_state.py_state[0].function_name, _domain_name.top().c_str());
          }
        }
        _delayed_trace->python_state = _state;
      }

      // STEP 2
      // Normal update on call_path_map
  # ifdef DEBUG
      std::cout << "We Got " <<  _trace->access_memory.size() << " memory accesses." << std::endl;
  # endif
      for (auto & [pc, m_c] : _trace->access_memory) {
        for (auto & [m, c] : m_c) {  // m is real mem_start addr, c is 0 place holder. Use op->ctx_id instead
          u64 mem_start = m;
          std::vector<ViewNode*> view_node_hit_mem;

          if(_pc_node_cache.find(pc) != _pc_node_cache.end()) {
            update_node_total_access(_pc_node_cache[pc], pc);
            continue; // just update access counter, but dont add callpath again and again
          } else {
            view_node_hit_mem = new_get_view_nodes_by_mem_addr(mem_start, false);
            update_node_total_access(view_node_hit_mem, pc);
            _pc_node_cache[pc] = view_node_hit_mem;
          }

          // Update Call ctc_id to CallPath
          for (auto viter = view_node_hit_mem.begin(); viter != view_node_hit_mem.end(); viter++){
            call_path_map[(*viter)->view_id].back().ctxid_pcs[op->ctx_id].push_back(pc);

            // if this is a aten::copy_ domain
            if (!_domain_name.empty()) {
              std::cout << "Domain: " << _domain_name.top() << std::endl;
            }

            // Detect Dead Write
            if(_trace->write_pcs.find(pc) != _trace->write_pcs.end()) {
              if ((*viter)->is_memory_write_pc) {
                write_after_write_map[(*viter)->view_id].emplace_back(aten_copy_pystate_hash, pc);
              }
            } else {
              (*viter)->is_memory_write_pc = false;
            }


            if (is_aten_copy_domain) {
              if (aten_copy_tar == 0 || aten_copy_src == 0) {
                // std::string all_states("");
                // if (num__delayed_states > 0 && num__delayed_states <= MAX_NUM_STATES) {
                //   for(size_t i = 0; i < num__delayed_states; i++) {
                //     all_states.append(delayed_python_states[i].file_name);
                //     all_states.append(delayed_python_states[i].function_name);
                //     all_states.append(std::to_string(delayed_python_states[i].function_first_lineno));
                //     all_states.append(std::to_string(delayed_python_states[i].lineno));
                //   }
                // }
                // // std::cout << "all_states: " << all_states << std::endl;
                // aten_copy_pystate_hash = (std::size_t)std::hash<std::string>{}(all_states);
                std::cout << "unit access: " << aten_copy_pystate_hash << std::endl;
                if (call_path_map[(*viter)->view_id].back().index == 0) {
                  aten_copy_tar = (*viter)->view_id;
                } else if (call_path_map[(*viter)->view_id].back().index == 1) {
                  aten_copy_src = (*viter)->view_id;
                }
                if(aten_copy_tar != 0 && aten_copy_src != 0) {
                  device_view_copy_map[aten_copy_pystate_hash][aten_copy_tar] = aten_copy_src;
                }
              }
            } // Set aten_copy_tar & aten_copy_src
          } // TODO need to be tested
  # ifdef DEBUG
          std::cout << "Kernel Access Hits: " << view_node_hit_mem.size() << " View Node(s). :: " << mem_start << std::endl;
  # endif
          if (view_node_hit_mem.empty()){
            // if (!_delayed_trace->access_memory.has(pc)) {
            if (true) {
              _delayed_trace->access_memory[pc].emplace(mem_start, op->ctx_id);
              if(_trace->write_pcs.find(pc) != _trace->write_pcs.end()) {
                _delayed_trace->write_pcs.insert(pc);
              }
            }
          }
        }
      }

  # ifdef DEBUG
      std::cout << "Will delay mem size: " << _delayed_trace->access_memory.size() << std::endl;
  # endif
      // check if any unit access has been delayed
      if(_delayed_trace->access_memory.empty()){
        _delayed_trace = NULL;
      }
      _trace->access_memory.clear();
      _trace->write_pcs.clear();
      _trace = NULL;
    }

    /**
     * On-GPU trace_data; global memory access mapping 
     * (Viewnode memory range)
     * read_pc_range_bit_map & write_pc_range_bit_map are 0/1 bit maps
     * For better understanding to the data strructure, 
     *    refer to gpu-patch.h/gpu_patch_aux_torchview_dict_t
     * 
     * PyTorch function input tensor typed objects' memory ranges and their ViewNode ptr are preprocessed and stored in torchview_memory_snapshot
     * 
     * 1. get range size
     * 2. extract global access pc(s): STG LDG 
     * 3. use bit mask to bridge pc and memory range
     * 4. map memory range to ViewNode*.
     */
    else {
      if (torchview_dict->view_range_size == 0){
        return;
      }
      // TENPROF: the GPU dict was sized from torchview_memory_snapshot at launch;
      // if the snapshot changed by analysis time the bitmap indices no longer line
      // up. Quietly skip this kernel's on-GPU analysis instead of asserting -- a
      // failed assert() would take the whole process down.
      if (torchview_dict->view_range_size != torchview_memory_snapshot.size()) {
        return;
      }

      uint64_t write_pc_size = torchview_dict->current_write_pc_size;
      uint64_t read_pc_size = torchview_dict->current_read_pc_size;
      uint64_t* read_pc_range_bit_map = torchview_dict->read_pc_range_bit_map;
      uint64_t* write_pc_range_bit_map = torchview_dict->write_pc_range_bit_map;

      gpu_patch_analysis_address_t* start_end = torchview_dict->start_end;
      // for (size_t i = 0; i < torchview_dict->view_range_size; i++) {
      //   printf("start: %p end: %p \n", start_end[i].start, start_end[i].end); 
      // }

      size_t map_column_size = torchview_memory_snapshot.size() / 64;
      for (size_t r_i = 0; r_i < read_pc_size; r_i++) {
        uint64_t pc = r_i * 8;
        // uint64_t pc = *(read_pc_range_bit_map + r_i * (map_column_size + 2) + map_column_size + 1);
        std::set<ViewNode*> view_node_hit_set = {};
        std::vector<ViewNode*> view_node_hit_mem;
        for (size_t r_j = 0; r_j <= map_column_size; r_j++) {
          uint64_t range_group = *(read_pc_range_bit_map + r_i * (map_column_size + 2) + r_j);
          if (range_group == 0) { // all 64 view range miss hit
            continue;
          } else { // At least 1 view range got hit by the pc
            uint64_t mask = 1;
            for (size_t shift_offset = 0; shift_offset < 64; shift_offset++) {
              uint64_t flag = (mask << shift_offset);
              if((range_group & flag) == flag) { // Hit! Start logging data
                size_t _adv = r_j * 64 + shift_offset;  // TENPROF: bound -- a set bit
                if (_adv >= torchview_memory_snapshot.size()) continue;  // in the partial
                std::map<redshow::MemoryRange, redshow::TorchView::ViewNode *>::iterator iter = torchview_memory_snapshot.begin(); // last column overshoots size() -> std::advance past end() was UB
                std::advance(iter, _adv);
                if(iter->second){
                  view_node_hit_set.insert(iter->second);
                }
              } // finish logging data
            }
          }
        }
        // start logging data
        copy(view_node_hit_set.begin(), view_node_hit_set.end(), back_inserter(view_node_hit_mem));
        update_node_total_access(view_node_hit_mem, pc);

        if(!view_node_hit_mem.empty()){
          std::cout << "call_path_map size: " << call_path_map.size() << std::endl;
        // Update Call ctc_id to CallPath
          for (auto viter = view_node_hit_mem.begin(); viter != view_node_hit_mem.end(); viter++){
            if (call_path_map.find((*viter)->view_id) == call_path_map.end()){
              continue;
            }
            call_path_map[(*viter)->view_id].back().ctxid_pcs[op->ctx_id].push_back(pc);

            // if this is a aten::copy_ domain
            if (!_domain_name.empty()) {
              std::cout << "Domain: " << _domain_name.top() << std::endl;
            }

            if (is_aten_copy_domain) {
              if (aten_copy_tar == 0 || aten_copy_src == 0) {
                std::cout << "unit access: " << aten_copy_pystate_hash << std::endl;
                if (call_path_map[(*viter)->view_id].back().index == 0) {
                  aten_copy_tar = (*viter)->view_id;
                } else if (call_path_map[(*viter)->view_id].back().index == 1) {
                  aten_copy_src = (*viter)->view_id;
                }
                if(aten_copy_tar != 0 && aten_copy_src != 0) {
                  device_view_copy_map[aten_copy_pystate_hash][aten_copy_tar] = aten_copy_src;
                }
              }
            } // Set aten_copy_tar & aten_copy_src
          }
        }
      }

      // Write map
      for (size_t r_i = 0; r_i < write_pc_size; r_i++) {
        // uint64_t pc = *(write_pc_range_bit_map + (r_i * 8) * (map_column_size + 2) + map_column_size + 1);
        uint64_t pc = r_i * 8;
        std::set<ViewNode*> view_node_hit_set = {};
        std::vector<ViewNode*> view_node_hit_mem;
        for (size_t r_j = 0; r_j <= map_column_size; r_j++) {
          uint64_t range_group = *(write_pc_range_bit_map + r_i * (map_column_size + 2) + r_j);
          if (range_group == 0) { // all 64 view range miss hit
            continue;
          } else { // At least 1 view range got hit by the pc
            uint64_t mask = 1;
            for (size_t shift_offset = 0; shift_offset < 64; shift_offset++) {
              uint64_t flag = (mask << shift_offset);
              if((range_group & flag) == flag) { // Hit! Start logging data
                size_t _adv = r_j * 64 + shift_offset;  // TENPROF: bound -- a set bit
                if (_adv >= torchview_memory_snapshot.size()) continue;  // in the partial
                std::map<redshow::MemoryRange, redshow::TorchView::ViewNode *>::iterator iter = torchview_memory_snapshot.begin(); // last column overshoots size() -> std::advance past end() was UB
                std::advance(iter, _adv);
                if(iter->second){
                  view_node_hit_set.insert(iter->second);
                }
              } // finish logging data
            }
          }
        }
        // start logging data
        copy(view_node_hit_set.begin(), view_node_hit_set.end(), back_inserter(view_node_hit_mem));
        update_node_total_access(view_node_hit_mem, pc);

        // Update Call ctc_id to CallPath
        if (!view_node_hit_mem.empty()){
          std::cout << "call_path_map size: " << call_path_map.size() << std::endl;
          for (auto viter = view_node_hit_mem.begin(); viter != view_node_hit_mem.end(); viter++){
            if (call_path_map.find((*viter)->view_id) == call_path_map.end()){
              continue;
            }
            call_path_map[(*viter)->view_id].back().ctxid_pcs[op->ctx_id].push_back(pc);


            if ((*viter)->is_memory_write_pc) {
                write_after_write_map[(*viter)->view_id].emplace_back(aten_copy_pystate_hash, pc);
            }
            (*viter)->is_memory_write_pc = true;

            // if this is a aten::copy_ domain
            if (!_domain_name.empty()) {
              std::cout << "Domain: " << _domain_name.top() << std::endl;
            }
            if (is_aten_copy_domain) {
              if (aten_copy_tar == 0 || aten_copy_src == 0) {
                std::cout << "unit access: " << aten_copy_pystate_hash << std::endl;
                if (call_path_map[(*viter)->view_id].back().index == 0) {
                  aten_copy_tar = (*viter)->view_id;
                } else if (call_path_map[(*viter)->view_id].back().index == 1) {
                  aten_copy_src = (*viter)->view_id;
                }
                if(aten_copy_tar != 0 && aten_copy_src != 0) {
                  device_view_copy_map[aten_copy_pystate_hash][aten_copy_tar] = aten_copy_src;
                }
              }
            } // Set aten_copy_tar & aten_copy_src
          }
        }
      }
    }
  }

  void TorchView::memcpy_op_callback(std::shared_ptr<Memcpy> op) {
    // TENPROF race fix: serialize on the SHARED torch_view mutex (no GIL acquired
    // here, so locking at the top is safe). Was implicitly under op_callback's
    // Analysis::_lock, a different mutex than torch_view_callback's.
    std::lock_guard<std::mutex> _tv_lk(redshow_torchview_mtx());
    if ((op->dst_memory_op_id != REDSHOW_MEMORY_HOST && op->dst_memory_op_id != REDSHOW_MEMORY_UVM) // im case dst on device
         ||
        (op->src_memory_op_id != REDSHOW_MEMORY_HOST && op->src_memory_op_id != REDSHOW_MEMORY_UVM)) { // in case src on device
      // auto dst_ctx_id = _op_node.at(op->dst_memory_op_id);
      u64 overwrite_len = op->len;
      u64 src_start = op->src_start;
      u64 dst_start = op->dst_start;
      u64 dst_shadow_start = op->dst_shadow_start;

      // (mem_range_t)mem_range{mem_addrs, mem_addrs + op->len};
      std::vector<ViewNode*> view_node_hit_src = {};// get_view_nodes_by_mem_addr(src_start);
      std::vector<ViewNode*> view_node_hit_dst = {};// get_view_nodes_by_mem_addr(dst_start);
      std::vector<ViewNode*> view_node_hit_shadow = {};//get_view_nodes_by_mem_addr(dst_shadow_start);

# ifdef DEBUG
      std::cout << "memcpy hit: " << view_node_hit_src.size() << " " <<
                                     view_node_hit_dst.size() << " " <<
                                     view_node_hit_shadow.size() << " view nodes." << std::endl;
# endif
      update_node_total_access(view_node_hit_src, 0);
      update_node_total_access(view_node_hit_dst, 0);
      update_node_total_access(view_node_hit_shadow, 0);

      // TODO update call_path_map
      if(!view_node_hit_src.empty()) {
        for (auto viter: view_node_hit_src) {
          call_path_map[(*viter).view_id].back().ctxid_pcs[op->ctx_id].push_back(0);
        }
      }
      if(!view_node_hit_dst.empty()) {
        for (auto viter: view_node_hit_dst) {
          call_path_map[(*viter).view_id].back().ctxid_pcs[op->ctx_id].push_back(0);
        }
      }
      if(!view_node_hit_shadow.empty()) {
        for (auto viter: view_node_hit_shadow) {
          call_path_map[(*viter).view_id].back().ctxid_pcs[op->ctx_id].push_back(0);
        }
      }
    }
  }

  void TorchView::memset_op_callback(std::shared_ptr<Memset> op) {
    // TENPROF race fix: serialize on the SHARED torch_view mutex (no GIL here).
    std::lock_guard<std::mutex> _tv_lk(redshow_torchview_mtx());
    if (op->memory_op_id != REDSHOW_MEMORY_HOST && op->memory_op_id != REDSHOW_MEMORY_UVM) {
      u64 overwrite = op->len;
      u64 start = op->start;
      u64 dst_shadow_start = op->shadow_start;
      u64 value = op->value;

      std::vector<ViewNode*> view_node_hit_start = {}; //get_view_nodes_by_mem_addr(start);
      std::vector<ViewNode*> view_node_hit_shadow = {}; //get_view_nodes_by_mem_addr(dst_shadow_start);

# ifdef DEBUG
      std::cout << "memset hit: " << view_node_hit_start.size() << " " <<
                                     view_node_hit_shadow.size() << " view nodes." << std::endl;
# endif
      update_node_total_access(view_node_hit_start, 0);
      update_node_total_access(view_node_hit_shadow, 0);

      if(!view_node_hit_start.empty()){
        for (auto viter : view_node_hit_start) {
          call_path_map[(*viter).view_id].back().ctxid_pcs[op->ctx_id].push_back(0);
        }
      }
      if(!view_node_hit_shadow.empty()){
        for (auto viter : view_node_hit_shadow) {
          call_path_map[(*viter).view_id].back().ctxid_pcs[op->ctx_id].push_back(0);
        }
      }
    }
  }

  void TorchView::analysis_begin(u32 cpu_thread, i32 kernel_id, u64 host_op_id, u32 stream_id,
                                u32 cubin_id, u32 mod_id, GPUPatchType type, void* trace_data) {
    assert(type == GPU_PATCH_TYPE_ADDRESS_PATCH || type == GPU_PATCH_TYPE_ADDRESS_ANALYSIS);
    // std::cout << "Redshow torchview->analysis_begin.." << std::endl;
    lock();
    if (!torch_view_ongpu) {
      if (!_trace) {
        _trace = std::make_shared<TorchViewTrace>();
      }
    } else { // torch_view_ongpu
      torchview_dict = (gpu_patch_aux_torchview_dict_t*)(static_cast<gpu_patch_buffer_t*>(trace_data))->aux;
      std::cout << "trace_data: torchview_dict " << torchview_dict << std::endl;
    }
    if (!_domain_name.empty() && _domain_name.top().compare("aten::copy_") == 0){
      is_aten_copy_domain = true;
      aten_copy_pystate_hash = 0;
      aten_copy_tar = 0;
      aten_copy_src = 0;
    }
# ifdef DEBUG
    std::cout << "analysis_begin Kernel ID: " << std::hex << kernel_id << std::dec << std::endl;
# endif
    unlock();
  }

  void TorchView::analysis_end(u32 cpu_thread, i32 kernel_id) {
# ifdef DEBUG
    std::cout << "analysis_end Kernel ID: " << std::hex << kernel_id << std::dec << std::endl;
# endif
    if (is_aten_copy_domain == true && aten_copy_src != 0 && aten_copy_src != 0) {
      is_aten_copy_domain = false;
      aten_copy_pystate_hash = 0;
      aten_copy_tar = 0;
      aten_copy_src = 0;
    }
  }

  void TorchView::block_enter(const ThreadId &thread_id) {
    // No operation
  }

  void TorchView::block_exit(const ThreadId &thread_id) {
    // No operation
  }

  void TorchView::unit_access(i32 kernel_id, u64 host_op_id, const ThreadId &thread_id,
                                 const AccessKind &access_kind, const Memory &memory, u64 pc,
                                 u64 value, u64 addr, u32 index, GPUPatchFlags flags) {
    // if (true) { 
# ifdef DEBUG
      std::cout << "pc: " << std::hex << pc << " mem: " << memory.memory_range.start << std::dec << std::endl;
# endif
    if (!_trace->access_memory.has(pc)) {
      _trace->access_memory[pc].emplace(memory.memory_range.start, 0); // 0 placeholder;
      if (static_cast<GPUPatchFlags>(flags) == GPU_PATCH_WRITE) {
        _trace->write_pcs.insert(pc);
      }
    }
  }

  void TorchView::flush_thread(u32 cpu_thread, const std::string &output_dir,
                                  const LockableMap<u32, Cubin> &cubins,
                                  redshow_record_data_callback_func record_data_callback) {}

  void TorchView::flush(const std::string &output_dir, const LockableMap<u32, Cubin> &cubins,
                        redshow_record_data_callback_func record_data_callback)
/**
 * new version
 * */
  {
    // clear delayed unit access trace
    auto t_start = std::chrono::system_clock::now();
    map_delayed_access();
    clear_delayed_trace();

    // lock();
    // Log device_view_copy_map
    std::ofstream copy_out(output_dir + "aten_copy_map.txt");
    copy_out << "Target   :   Source" << std::endl;
    for(auto& [_hash, t_s] : device_view_copy_map) {
      copy_out << _hash << std::endl;
      for(auto& [tar, src] : t_s) {
        copy_out << "        " << tar << " : " << src << std::endl;
      }
    }

    copy_out << std::endl;
    copy_out << "Write After Write Map" << std::endl;
    for(auto& [view_id, pcs] : write_after_write_map) {
      copy_out << "View ID: " << view_id << std::endl;
      for(auto& [hash, pc] : pcs) {
        copy_out << "    " << hash << " : " << pc << std::endl;
      }
    }
    copy_out.close();


/**
  * Log the View forest 
  * 
  */
    // Log the forest
    if (!forest_tree_out.is_open()){
      forest_tree_out = std::ofstream(output_dir + "forest.txt", std::ios::app);
    }
    // std::ofstream fout(output_dir + "forest.txt", std::ios::app);
    for (unsigned i = 0; i < _roots.size(); i++) {
      if (_roots.at(i)->_children.empty() && call_path_map[_roots.at(i)->view_id].at(0).ctxid_pcs.empty()) {
        auto _dead = call_path_map.find(_roots.at(i)->view_id);
        if (_dead != call_path_map.end()) {
          call_path_map.erase(_dead);
        }
      } else {
        _roots.at(i)->delete_children_nodes(forest_tree_out, viewnode_mem_range_map);
        forest_tree_out << '\n';
      }
      delete _roots.at(i);
      _roots.erase(_roots.begin()+i);
      --i;
    }
    // delete_forest_tree(output_dir, 0, ((uint64_t)0) - 1);
    forest_tree_out.close();

/**
  * Log ViewNode access info
  * 
  */
    std::ofstream out(output_dir + "torch_view_report.csv");

    // REMOVE
    // END REMOVE
    for(auto iter = call_path_map.begin(); iter != call_path_map.end(); iter++){
      out << "id " << iter->first << std::endl;
      out << "python_state " << std::endl; // Python StateS begin
      for(auto siter = iter->second.begin(); siter != iter->second.end(); siter++){
        // skip states with no ctx info
        // if(siter->ctxid_pcs.empty()) {
        //   continue;
        // }
        // end skip states with no ctx info
        out << "index " << siter->index << std::endl;
        out << "num_states " << siter->num_states << std::endl;
        out << "py_state " << std::endl; // py_state begin
        size_t state_length = (siter->num_states < MAX_NUM_STATES ? siter->num_states : MAX_NUM_STATES);
        for(size_t i = 0; i < state_length; i++) {
          out << "file_name "<< siter->py_state[i].file_name << std::endl;
          out << "function_name "<< siter->py_state[i].function_name << std::endl;
          out << "function_first_lineno "<< siter->py_state[i].function_first_lineno << std::endl;
          out << "lineno "<< siter->py_state[i].lineno << std::endl;
        }
       // start
        std::string fitst_filename;
        std::string all_states("");
        if (state_length > 0) {
          const size_t last_slash_idx = std::string(siter->py_state[0].function_name).rfind('^');
          if (std::string::npos != last_slash_idx) {
            fitst_filename = std::string(siter->py_state[0].function_name).substr(0, last_slash_idx);
          }
        }
        for(size_t i = 0; i < state_length; i++) {
          all_states.append(siter->py_state[i].file_name);
          if (i == 0 && fitst_filename.length() > 0) {
            all_states.append(fitst_filename);
          } else {
            all_states.append(siter->py_state[i].function_name);
          }
          all_states.append(std::to_string(siter->py_state[i].function_first_lineno));
          all_states.append(std::to_string(siter->py_state[i].lineno));
        }
        out << "pytates_hash " << (std::size_t)std::hash<std::string>{}(all_states) << std::endl;
        // end
        out << "object_type " << (int)siter->object_type << std::endl;
        for(auto& [ctx_id, pc_s] : siter->ctxid_pcs){
          // out << "ctx_id " << std::endl; // ctx_id begin
          // out << ctx_id << std::endl;
          for(auto& pc : pc_s) {
            out << "ctx_id " << std::endl; // ctx_id begin
            out << ctx_id << std::endl;
            out << "pc " << std::endl;
            out << pc << std::endl;
          }
        }
        out << std::endl;
      }
    }
    auto t_end = std::chrono::system_clock::now();
    auto ms_int = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start);
    std::cout << "Dump Data takes: " << ms_int.count() << std::endl;

    out.close();

    // unlock();
  }

}  // namespace redshow
