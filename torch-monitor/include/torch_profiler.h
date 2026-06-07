#ifndef TORCH_MONITOR_TORCH_PROFILER_H
#define TORCH_MONITOR_TORCH_PROFILER_H

#include <unordered_set>

#include "torch_monitor.h"
#include "utils.h"

namespace torch_monitor {

class TorchProfiler {
 public:
  void enable_memory_profiling() {
    _is_memory_profiling_enabled = true;
  }

  void disable_memory_profiling() {
    _is_memory_profiling_enabled = false;
  }

  bool is_memory_profiling_enabled() {
    return _is_memory_profiling_enabled;
  }

  // true: register success
  // false: register fail
  bool register_domain(torch_monitor_domain_t domain);

  // true: domain registered
  // false: domain not registered
  bool has_domain(torch_monitor_domain_t domain);

  // true: register success
  // false: register fail
  bool register_callback(torch_monitor_callback_func_t callback);

  // true: start profiling
  // false: cannot start profiling
  bool start_profiling();

  // true: stop profiling
  // false: cannot stop profiling
  bool stop_profiling();

  // true: start profiling
  // false: cannot start profiling
  bool start_memory_profiling();

  // true: stop profiling
  // false: cannot stop profiling
  bool stop_memory_profiling();

  // Get the singleton instance
  static TorchProfiler& instance();

 public:
  const static int64_t TORCH_PROFILER_SEQUENCE_NUMBER_NULL = -1;
  const static int64_t TORCH_PROFILER_HANDLE_NULL = 0;

 private:
  TorchProfiler() {
  }

  class MemoryState : public c10::MemoryReportingInfoBase {
   public:
    MemoryState() {
    }

    bool memoryProfilingEnabled() const override {
      return true;
    }

    // Memory allocatation callback
#if TORCH_VERSION_MAJOR >= 2
    void reportMemoryUsage(void* ptr, int64_t alloc_size, size_t total_allocated,
                           size_t total_reserved, c10::Device device) override;
#else
    void reportMemoryUsage(void* ptr, int64_t alloc_size, int64_t total_allocated,
                           int64_t total_reserved, c10::Device device) override;
#endif
  };

  // Generate a memory state for every thread
  std::shared_ptr<MemoryState> new_memory_state() {
    if (_is_memory_profiling_enabled) {
      return std::make_shared<MemoryState>();
    } else {
      return nullptr;
    }
  }

  // true: init success
  // false: init fail
  static bool init_callback_data(torch_monitor_callback_site_t callback_site,
                                 const at::RecordFunction& fn,
                                 torch_monitor_callback_data_t& callback_data);

 private:
  bool _is_memory_profiling_enabled = false;
};

}  // namespace torch_monitor

#endif  // TORCH_MONITOR_TORCH_PROFILER_H
