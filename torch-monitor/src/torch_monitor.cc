#include "torch_monitor.h"

#include "python_state.h"
#include "torch_profiler.h"
#include "utils.h"

namespace torch_monitor {

EXTERNC torch_monitor_status_t
torch_monitor_callback_subscribe(torch_monitor_callback_func_t func) {
  LOG_INFO("Enter torch_monitor_callback_subscribe");

  torch_monitor_status_t status;

  auto& profiler = TorchProfiler::instance();

  if (func) {
    if (profiler.register_callback(func)) {
      status = TORCH_MONITOR_STATUS_SUCCESS;
    } else {
      status = TORCH_MONITOR_STATUS_SUBSCRIBE_EXIST;
    }
  } else {
    status = TORCH_MONITOR_STATUS_SUBSCRIBE_SUBSCRIBER_NULL;
  }

  LOG_INFO("Exit torch_monitor_callback_subscribe");
  return status;
}

EXTERNC torch_monitor_status_t
torch_monitor_domain_enable(torch_monitor_domain_t domain) {
  LOG_INFO("Enter torch_monitor_domain_enable");

  torch_monitor_status_t status;

  auto& profiler = TorchProfiler::instance();

  if (domain < TORCH_MONITOR_DOMAIN_COUNT && profiler.register_domain(domain)) {
    status = TORCH_MONITOR_STATUS_SUCCESS;
  } else {
    status = TORCH_MONITOR_STATUS_ENABLE_DOMAIN_OUT_RANGE;
  }

  LOG_INFO("Exit torch_monitor_domain_enable");
  return status;
}

/* ===== ORIGINAL torch_monitor_init() — commented out, see replacement below =====
EXTERNC torch_monitor_status_t torch_monitor_init() {
  LOG_INFO("Enter torch_monitor_init");

  torch_monitor_status_t status;

  auto& profiler = TorchProfiler::instance();

  if (profiler.start_profiling()) {
    status = TORCH_MONITOR_STATUS_SUCCESS;
  } else {
    status = TORCH_MONITOR_STATUS_INIT_HANDLE_FAIL;
  }

  torch_monitor_thread_init();

  LOG_INFO("Exit torch_monitor_init");
  return status;
}
===== END ORIGINAL torch_monitor_init() ===== */

// Replacement: do NOT call torch_monitor_thread_init() here. That ran
// c10::ThreadLocalDebugInfo::_push() at init time, which can execute during the
// dynamic loader's startup (hpcrun intercepting cublasLt's first dlopen), before
// torch/c10 is initialized -> SIGSEGV. The per-thread memory state is now pushed
// lazily on the first RecordFunction callback (see TorchProfiler::start_profiling).
EXTERNC torch_monitor_status_t torch_monitor_init() {
  LOG_INFO("Enter torch_monitor_init");

  torch_monitor_status_t status;

  auto& profiler = TorchProfiler::instance();

  if (profiler.start_profiling()) {
    status = TORCH_MONITOR_STATUS_SUCCESS;
  } else {
    status = TORCH_MONITOR_STATUS_INIT_HANDLE_FAIL;
  }

  LOG_INFO("Exit torch_monitor_init");
  return status;
}

EXTERNC torch_monitor_status_t torch_monitor_thread_init() {
  LOG_INFO("Enter torch_monitor_thread_init");

  torch_monitor_status_t status;

  auto& profiler = TorchProfiler::instance();

  if (profiler.start_memory_profiling()) {
    status = TORCH_MONITOR_STATUS_SUCCESS;
  } else {
    status = TORCH_MONITOR_STATUS_INIT_MEMORY_NOT_REGISTER;
  }

  LOG_INFO("Exit torch_monitor_thread_init");
  return status;
}

EXTERNC torch_monitor_status_t torch_monitor_finalize() {
  LOG_INFO("Enter torch_monitor_finalize");

  torch_monitor_status_t status;

  auto& profiler = TorchProfiler::instance();

  if (profiler.stop_profiling()) {
    status = TORCH_MONITOR_STATUS_SUCCESS;
  } else {
    status = TORCH_MONITOR_STATUS_FINALIZE_NOT_INIT;
  }

  LOG_INFO("Exit torch_monitor_finalize");
  return status;
}

EXTERNC torch_monitor_status_t torch_monitor_thread_finalize() {
  LOG_INFO("Enter torch_monitor_thread_finalize");

  torch_monitor_status_t status;

  auto& profiler = TorchProfiler::instance();

  if (profiler.stop_memory_profiling()) {
    status = TORCH_MONITOR_STATUS_SUCCESS;
  } else {
    status = TORCH_MONITOR_STATUS_FINALIZE_MEMORY_FAIL;
  }

  LOG_INFO("Exit torch_monitor_thread_finalize");

  return status;
}

EXTERNC torch_monitor_status_t torch_monitor_python_state_get(
    size_t max_num_states, torch_monitor_python_state_t* states, size_t* num_states) {
  LOG_INFO("Enter torch_monitor_python_state_get");

  torch_monitor_status_t status;

  auto& python_state_monitor = PythonStateMonitor::instance();

  auto& python_states = python_state_monitor.get_states();

  if (python_states.empty()) {
    status = TORCH_MONITOR_STATUS_PYTHON_STATES_NULL;
  } else {
    status = TORCH_MONITOR_STATUS_SUCCESS;

    *num_states = std::min(python_states.size(), max_num_states);
    for (size_t i = 0; i < *num_states; ++i) {
      states[i].file_name = python_states[i].file_name.c_str();
      states[i].function_name = python_states[i].function_name.c_str();
      states[i].function_first_lineno = python_states[i].function_first_lineno;
      states[i].lineno = python_states[i].lineno;
    }
  }

  LOG_INFO("Exit torch_monitor_python_state_get");

  return status;
}

// TENPROF: cached variant -- returns the thread-local Python state from the most
// recent fresh fetch (the launching op's enter) WITHOUT acquiring the GIL. Used by
// redshow's kernel_op_callback, which runs while the sanitizer holds the per-context
// entry->lock; acquiring the GIL there deadlocks against a thread that holds the GIL
// and spins on entry->lock (AB-BA). Same copy-out as above, but get_states(true).
EXTERNC torch_monitor_status_t torch_monitor_python_state_get_cached(
    size_t max_num_states, torch_monitor_python_state_t* states, size_t* num_states) {
  torch_monitor_status_t status;

  auto& python_state_monitor = PythonStateMonitor::instance();

  auto& python_states = python_state_monitor.get_states(true);  // cached -> NO GIL

  if (python_states.empty()) {
    status = TORCH_MONITOR_STATUS_PYTHON_STATES_NULL;
  } else {
    status = TORCH_MONITOR_STATUS_SUCCESS;

    *num_states = std::min(python_states.size(), max_num_states);
    for (size_t i = 0; i < *num_states; ++i) {
      states[i].file_name = python_states[i].file_name.c_str();
      states[i].function_name = python_states[i].function_name.c_str();
      states[i].function_first_lineno = python_states[i].function_first_lineno;
      states[i].lineno = python_states[i].lineno;
    }
  }

  return status;
}

}  // namespace torch_monitor

static volatile bool torch_monitor_inputs_capture_enable = false;
static volatile bool torch_monitor_outputs_capture_enable = false;

EXTERNC bool torch_monitor_inputs_capture_enable_get() {
  return torch_monitor_inputs_capture_enable;
}

EXTERNC void torch_monitor_inputs_capture_enable_set(bool val) {
  torch_monitor_inputs_capture_enable = val;
}

EXTERNC bool torch_monitor_outputs_capture_enable_get() {
  return torch_monitor_outputs_capture_enable;
}

EXTERNC void torch_monitor_outputs_capture_enable_set(bool val) {
  torch_monitor_outputs_capture_enable = val;
}

const char* const dtype_name[] = {
    [TORCH_MONITOR_SCALAR_TYPES_BYTE] = "Byte",
    [TORCH_MONITOR_SCALAR_TYPES_CHAR] = "Char",
    [TORCH_MONITOR_SCALAR_TYPES_SHORT] = "Short",
    [TORCH_MONITOR_SCALAR_TYPES_INT] = "Int",
    [TORCH_MONITOR_SCALAR_TYPES_LONG] = "Long",
    [TORCH_MONITOR_SCALAR_TYPES_HALF] = "Half",
    [TORCH_MONITOR_SCALAR_TYPES_FLOAT] = "Float",
    [TORCH_MONITOR_SCALAR_TYPES_DOUBLE] = "Double",
    [TORCH_MONITOR_SCALAR_TYPES_COMPLEXHALF] = "ComplexHalf",
    [TORCH_MONITOR_SCALAR_TYPES_COMPLEXFLOAT] = "ComplexFloat",
    [TORCH_MONITOR_SCALAR_TYPES_COMPLEXDOUBLE] = "ComplexDouble",
    [TORCH_MONITOR_SCALAR_TYPES_BOOL] = "Bool",
    [TORCH_MONITOR_SCALAR_TYPES_QINT8] = "QInt8",
    [TORCH_MONITOR_SCALAR_TYPES_QUINT8] = "QUInt8",
    [TORCH_MONITOR_SCALAR_TYPES_QINT32] = "QInt32",
    [TORCH_MONITOR_SCALAR_TYPES_BFLOAT16] = "BFloat16",
    [TORCH_MONITOR_SCALAR_TYPES_QUINT4X2] = "QUInt4x2",
    [TORCH_MONITOR_SCALAR_TYPES_QUINT2X4] = "QUInt2x4",
    [TORCH_MONITOR_SCALAR_TYPES_BITS1X8] = "Bits1x8",
    [TORCH_MONITOR_SCALAR_TYPES_BITS2X4] = "Bits2x4",
    [TORCH_MONITOR_SCALAR_TYPES_BITS4X2] = "Bits4x2",
    [TORCH_MONITOR_SCALAR_TYPES_BITS8] = "Bits8",
    [TORCH_MONITOR_SCALAR_TYPES_BITS16] = "Bits16",
    [TORCH_MONITOR_SCALAR_TYPES_FLOAT8_E5M2] = "Float8_e5m2",
    [TORCH_MONITOR_SCALAR_TYPES_FLOAT8_E4M3FN] = "Float8_e4m3fn",
    [TORCH_MONITOR_SCALAR_TYPES_FLOAT8_E5M2FNUZ] = "Float8_e5m2fnuz",
    [TORCH_MONITOR_SCALAR_TYPES_FLOAT8_E4M3FNUZ] = "Float8_e4m3fnuz",
    [TORCH_MONITOR_SCALAR_TYPES_UNMATCHED_TYPE] = "Unmatched Type",
};

EXTERNC const char* torch_monitor_dtype_name_get(
    torch_monitor_scalar_type_t scalar_type) {
  return dtype_name[scalar_type];
}
