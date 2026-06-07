#include "torch_profiler.h"

#include "utils.h"
#include <ATen/core/ivalue.h>

namespace torch_monitor {

// A global state variable
// Since Aten record function does not capture anything,
// TorchProfilerState must be a static variable
struct TorchProfilerState {
  std::unordered_set<at::RecordScope, std::hash<at::RecordScope>> scopes;

  at::CallbackHandle handle = TorchProfiler::TORCH_PROFILER_HANDLE_NULL;

  torch_monitor_callback_func_t callback = nullptr;

  void clear() {
    callback = nullptr;
    handle = TorchProfiler::TORCH_PROFILER_HANDLE_NULL;
    this->scopes.clear();
  }

  static TorchProfilerState& instance() {
    static TorchProfilerState state;
    return state;
  }

 private:
  TorchProfilerState() {
  }
};

void store_tensor_info(torch_monitor_callback_data_t& callback_data, c10::IValue& ele,
                       int i) {
  auto tensor = ele.toTensor();
  callback_data.data.op_data.input_output_data.tensor_data[i].numel =
      (int64_t)tensor.numel();
  callback_data.data.op_data.input_output_data.tensor_data[i].dim = (int64_t)tensor.dim();
  if (callback_data.data.op_data.input_output_data.tensor_data[i].numel > 0) {
    callback_data.data.op_data.input_output_data.tensor_data[i].index = i;
    callback_data.data.op_data.input_output_data.tensor_data[i].dtype =
        (torch_monitor_scalar_type_t)tensor.dtype().toScalarType();
    callback_data.data.op_data.input_output_data.tensor_data[i].itemsize =
        (torch_monitor_scalar_type_t)tensor.dtype().itemsize();
    callback_data.data.op_data.input_output_data.tensor_data[i].data_ptr =
        tensor.data_ptr();
    callback_data.data.op_data.input_output_data.tensor_data[i].storage_offset =
        tensor.storage_offset();
    int64_t tensor_dim = std::min<int64_t>(
        TORCH_MONITOR_MAX_TENSOR_DIMENSION,
        callback_data.data.op_data.input_output_data.tensor_data[i].dim);
    c10::IntArrayRef size = tensor.sizes();
    c10::IntArrayRef strides = tensor.strides();
    for (int64_t j = 0; j < tensor_dim; j++) {
      callback_data.data.op_data.input_output_data.tensor_data[i].sizes[j] =
          (int64_t)size[j];
      callback_data.data.op_data.input_output_data.tensor_data[i].strides[j] =
          (int64_t)strides[j];
    }
  }  // aten functions allow arguments to be c10::optional[Tensor], in which
     // cases Tensor(is a place holder) will have 0 element and no data_ptr.
  callback_data.data.op_data.input_output_data.tensor_data[i].metadata_ptr =
      tensor.getIntrusivePtr().get();
}

void store_inputs_outputs_info(torch_monitor_callback_data_t& callback_data,
                               const at::RecordFunction& fn,
                               torch_monitor_callback_site_t callback_site) {
  callback_data.data.op_data.input_output_data.size =
      (callback_site == TORCH_MONITOR_CALLBACK_EXIT) ? fn.num_outputs() : fn.num_inputs();
  if (callback_data.data.op_data.input_output_data.size > 0) {
    int64_t size = std::min<int64_t>(callback_data.data.op_data.input_output_data.size,
                                     TORCH_MONITOR_MAX_INPUTS_OUTPUTS_SIZE);
    if (callback_site == TORCH_MONITOR_CALLBACK_EXIT) {
      const std::vector<c10::IValue> values = fn.outputs();
      for (int i = 0; i < size; i++) {
        callback_data.data.op_data.input_output_data.tensor_data[i].index = -1;
        c10::IValue ele = values[i];
        if (ele.isTensor()) {
          store_tensor_info(callback_data, ele, i);
        }
      }
    } else if (callback_site == TORCH_MONITOR_CALLBACK_ENTER) {
      c10::ArrayRef<const c10::IValue> values = fn.inputs();
      for (int i = 0; i < size; i++) {
        callback_data.data.op_data.input_output_data.tensor_data[i].index = -1;
        c10::IValue ele = values[i];
        if (ele.isTensor()) {
          store_tensor_info(callback_data, ele, i);
        }
      }
    }
  }
}

#if TORCH_VERSION_MAJOR >= 2
void TorchProfiler::MemoryState::reportMemoryUsage(void* ptr, int64_t alloc_size,
                                                   size_t total_allocated,
                                                   size_t total_reserved,
                                                   c10::Device device) {
#else
void TorchProfiler::MemoryState::reportMemoryUsage(void* ptr, int64_t alloc_size,
                                                   int64_t total_allocated,
                                                   int64_t total_reserved,
                                                   c10::Device device) {
#endif
  LOG_INFO("ptr: %p", ptr);
  LOG_INFO("alloc_size: %ld", alloc_size);
  LOG_INFO("total_allocated: %lu", total_allocated);
  LOG_INFO("total_reserved: %lu", total_reserved);

  torch_monitor_callback_data_t callback_data;
  callback_data.domain = TORCH_MONITOR_DOMAIN_MEMORY;
  callback_data.current_thread_id = at::RecordFunction::currentThreadId();
  callback_data.data.mem_data.type =
      alloc_size < 0 ? TORCH_MONITOR_MEM_DATA_FREE : TORCH_MONITOR_MEM_DATA_ALLOC;
  callback_data.data.mem_data.device_type = aten_device_type_match(device.type());
  callback_data.data.mem_data.ptr = ptr;
  callback_data.data.mem_data.size = alloc_size < 0 ? -alloc_size : alloc_size;
  callback_data.data.mem_data.total_allocated = total_allocated;
  callback_data.data.mem_data.total_reserved = total_reserved;

  auto& instance = TorchProfilerState::instance();
  instance.callback(TORCH_MONITOR_CALLBACK_ENTER, &callback_data);
}

bool TorchProfiler::init_callback_data(torch_monitor_callback_site_t callback_site,
                                       const at::RecordFunction& fn,
                                       torch_monitor_callback_data_t& callback_data) {
  static thread_local uint32_t nested_level = 0;
  if (callback_site == TORCH_MONITOR_CALLBACK_EXIT) {
    --nested_level;
  }

  LOG_INFO("thread_id: %lu", fn.threadId());
  LOG_INFO("forward_thread_id: %lu", fn.forwardThreadId());
  LOG_INFO("scope: %u", static_cast<unsigned int>(fn.scope()));
  LOG_INFO("async: %u", fn.isAsync());
  LOG_INFO("active: %u", fn.isActive());
  LOG_INFO("sequence_number: %ld", fn.seqNr());
  LOG_INFO("logical_thread_id: %lu", at::RecordFunction::currentThreadId());
  LOG_INFO("level: %u", nested_level);
#if TORCH_VERSION_MAJOR <= 1 && TORCH_VERSION_MINOR < 11
  LOG_INFO("name: %s", fn.name().str());
#else
  LOG_INFO("name: %s", fn.name());
#endif

  // seqNr == TORCH_PROFILER_SEQUENCE_NUMBER_NULL means this op is not
  // associated with a backprop op if (fn.seqNr() ==
  // TORCH_PROFILER_SEQUENCE_NUMBER_NULL) {
  //   return false;
  // }

  auto domain = aten_scope_match(fn.scope());
  if (domain == TORCH_MONITOR_DOMAIN_COUNT) {
    return false;
  }

  callback_data.domain = domain;
  callback_data.current_thread_id = at::RecordFunction::currentThreadId();
  callback_data.data.op_data.forward_thread_id = fn.forwardThreadId();
  callback_data.data.op_data.sequence_number = fn.seqNr();
  callback_data.data.op_data.nested_level = nested_level;
#if TORCH_VERSION_MAJOR <= 1 && TORCH_VERSION_MINOR < 11
  callback_data.data.op_data.name = fn.name().str();
#else
  callback_data.data.op_data.name = fn.name();
#endif

  if (callback_site == TORCH_MONITOR_CALLBACK_EXIT) {
    if (torch_monitor_outputs_capture_enable_get()) {
      store_inputs_outputs_info(callback_data, fn, callback_site);
    }
  }

  if (callback_site == TORCH_MONITOR_CALLBACK_ENTER) {
    ++nested_level;
    if (torch_monitor_inputs_capture_enable_get()) {
      store_inputs_outputs_info(callback_data, fn, callback_site);
    }
  }

  return true;
}

TorchProfiler& TorchProfiler::instance() {
  static TorchProfiler profiler;
  return profiler;
}

// True: if a domain is registered
// False: if a domain is not registered
bool TorchProfiler::has_domain(torch_monitor_domain_t domain) {
  if (domain == TORCH_MONITOR_DOMAIN_MEMORY) {
    return is_memory_profiling_enabled();
  } else {
    at::RecordScope scope = torch_monitor_domain_match(domain);
    if (scope == at::RecordScope::NUM_SCOPES) {
      return false;
    }
    auto& instance = TorchProfilerState::instance();
    return instance.scopes.find(scope) != instance.scopes.end();
  }
}

// True: register success
// False: register fail
bool TorchProfiler::register_domain(torch_monitor_domain_t domain) {
  if (domain == TORCH_MONITOR_DOMAIN_MEMORY) {
    enable_memory_profiling();
    return true;
  } else {
    at::RecordScope scope = torch_monitor_domain_match(domain);
    if (scope == at::RecordScope::NUM_SCOPES) {
      return false;
    }
    TorchProfilerState::instance().scopes.insert(scope);
    return true;
  }
}

// True: register success
// False: register fail
bool TorchProfiler::register_callback(torch_monitor_callback_func_t callback) {
  auto& instance = TorchProfilerState::instance();
  if (instance.callback == nullptr) {
    instance.callback = callback;
    return true;
  }
  return false;
}

/* ===== ORIGINAL start_profiling() — commented out, see replacement below =====
bool TorchProfiler::start_profiling() {
  auto& instance = TorchProfilerState::instance();
  if (instance.callback == nullptr || instance.scopes.empty()) {
    return false;
  }

  auto handle = at::addGlobalCallback(
      at::RecordFunctionCallback(
          [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
            LOG_INFO("Enter function");

            torch_monitor_callback_data_t callback_data = {};
            if (init_callback_data(TORCH_MONITOR_CALLBACK_ENTER, fn, callback_data)) {
              TorchProfilerState::instance().callback(TORCH_MONITOR_CALLBACK_ENTER,
                                                      &callback_data);
            }
            return nullptr;
          },
          [](const at::RecordFunction& fn, at::ObserverContext* ctx_ptr) {
            torch_monitor_callback_data_t callback_data = {};
            if (init_callback_data(TORCH_MONITOR_CALLBACK_EXIT, fn, callback_data)) {
              TorchProfilerState::instance().callback(TORCH_MONITOR_CALLBACK_EXIT,
                                                      &callback_data);
            }
            LOG_INFO("Exit function");
            return;
          })
          .needsInputs(torch_monitor_inputs_capture_enable_get())
          .needsOutputs(torch_monitor_outputs_capture_enable_get())
          .scopes(TorchProfilerState::instance().scopes));

  if (handle != TORCH_PROFILER_HANDLE_NULL) {
    instance.handle = handle;
    return true;
  }

  return false;
}
===== END ORIGINAL start_profiling() ===== */

// Replacement: identical to the original, except the ENTER callback now lazily
// pushes the per-thread memory state on the first real torch op (when c10/torch
// is guaranteed initialized). This replaces the init-time _push that crashed
// during dynamic-loader startup (see torch_monitor_init in torch_monitor.cc).
bool TorchProfiler::start_profiling() {
  auto& instance = TorchProfilerState::instance();
  if (instance.callback == nullptr || instance.scopes.empty()) {
    return false;
  }

  auto handle = at::addGlobalCallback(
      at::RecordFunctionCallback(
          [](const at::RecordFunction& fn) -> std::unique_ptr<at::ObserverContext> {
            LOG_INFO("Enter function");

            // Lazy, once-per-thread memory-state registration. start_memory_profiling()
            // performs c10::ThreadLocalDebugInfo::_push() and self-guards on the MEMORY
            // domain, so this is a no-op when memory profiling is disabled.
            static thread_local bool mem_state_pushed = false;
            if (!mem_state_pushed) {
              TorchProfiler::instance().start_memory_profiling();
              mem_state_pushed = true;
            }

            torch_monitor_callback_data_t callback_data = {};
            if (init_callback_data(TORCH_MONITOR_CALLBACK_ENTER, fn, callback_data)) {
              TorchProfilerState::instance().callback(TORCH_MONITOR_CALLBACK_ENTER,
                                                      &callback_data);
            }
            return nullptr;
          },
          [](const at::RecordFunction& fn, at::ObserverContext* ctx_ptr) {
            torch_monitor_callback_data_t callback_data = {};
            if (init_callback_data(TORCH_MONITOR_CALLBACK_EXIT, fn, callback_data)) {
              TorchProfilerState::instance().callback(TORCH_MONITOR_CALLBACK_EXIT,
                                                      &callback_data);
            }
            LOG_INFO("Exit function");
            return;
          })
          .needsInputs(torch_monitor_inputs_capture_enable_get())
          .needsOutputs(torch_monitor_outputs_capture_enable_get())
          .scopes(TorchProfilerState::instance().scopes));

  if (handle != TORCH_PROFILER_HANDLE_NULL) {
    instance.handle = handle;
    return true;
  }

  return false;
}

bool TorchProfiler::stop_profiling() {
  TorchProfilerState::instance().clear();
  return true;
}

bool TorchProfiler::start_memory_profiling() {
  if (has_domain(TORCH_MONITOR_DOMAIN_MEMORY)) {
    // Register the profiler to thread local state
    // c10 only has a ThreadLocalDebugInfo structure without a global structure
    auto mem_state_ptr = new_memory_state();
    c10::ThreadLocalDebugInfo::_push(c10::DebugInfoKind::PROFILER_STATE, mem_state_ptr);
    return true;
  } else {
    return false;
  }
}

bool TorchProfiler::stop_memory_profiling() {
  if (is_memory_profiling_enabled()) {
    // XXX(Keren): torch monitor cannot be used together with kineto
    // Both register the profiler_state to ThreadLocalDebugInfo
    if (c10::ThreadLocalDebugInfo::_pop(c10::DebugInfoKind::PROFILER_STATE) != nullptr) {
      disable_memory_profiling();
      return true;
    } else {
      return false;
    }
  }
  return true;
}

}  // namespace torch_monitor
