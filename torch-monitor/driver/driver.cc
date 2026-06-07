#include <torch_monitor.h>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>

#define TORCH_MONITOR_CALL(func, args)                              \
  do {                                                              \
    torch_monitor_status status = func args;                        \
    if (status != TORCH_MONITOR_STATUS_SUCCESS) {                   \
      std::cerr << "Torch monitor status: " << status << std::endl; \
      exit(1);                                                      \
    }                                                               \
  } while (0)

// If python call path is enabled
volatile static bool python_state_enable = false;
// If timestamp is obtained
volatile static bool timestamp_enable = false;
// If the gdb is attached for GDB
volatile static bool driver_debug = false;
// If callback data are printed out
volatile static bool verbose = true;
// Maximum number of call path frames
const static size_t MAX_NUM_STATES = 30;
// Call path buffer
thread_local static torch_monitor_python_state_t python_states[MAX_NUM_STATES];
// Mutex for printing
static std::mutex mtx;

static void callback_inputs_outputs_report(torch_monitor_callback_site_t callback_site,
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
    std::cout << "    Tensor item size: " << (std::uint64_t)tensor.itemsize << std::endl;
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
    std::cout << "    Tensor block address: " << std::hex << tensor.data_ptr << std::dec
              << std::endl;
    std::cout << "    Intrusive Ptr: " << std::hex << tensor.metadata_ptr << std::dec
              << std::endl;
  }
}

static void python_state_report() {
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
}

static void driver_callback(torch_monitor_callback_site_t callback_site,
                            torch_monitor_callback_data_t* callback_data) {
  // If debug is enabled, hanging there and invoke GDB
  while (driver_debug) {
  }

  // If verbose is disabled, do not print anything.
  // It can be used to measure the wrapping overhead brought by torch-monitor
  if (!verbose) {
    return;
  }

  std::lock_guard<std::mutex> lock_guard(mtx);
  std::cout << "-----------------------------------" << std::endl;
  if (callback_site == TORCH_MONITOR_CALLBACK_ENTER) {
    std::cout << "Enter Domain: " << callback_data->domain << std::endl;
    if (callback_data->domain != TORCH_MONITOR_DOMAIN_MEMORY) {
      std::cout << "Current thread id: " << callback_data->current_thread_id << std::endl;
      std::cout << "Forward thread id: " << callback_data->data.op_data.forward_thread_id
                << std::endl;
      std::cout << "Sequence number: " << callback_data->data.op_data.sequence_number
                << std::endl;
      std::cout << "Name: " << std::string(callback_data->data.op_data.name) << std::endl;
      if (timestamp_enable) {
        std::cout << "Enter level: " << callback_data->data.op_data.nested_level << " at "
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count()
                  << std::endl;
      }
      if (python_state_enable) {
        python_state_report();
      }
      if (torch_monitor_inputs_capture_enable_get()) {
        callback_inputs_outputs_report(callback_site, callback_data);
      }
    } else {
      std::cout << "Current thread id: " << callback_data->current_thread_id << std::endl;
      if (callback_data->data.mem_data.type == TORCH_MONITOR_MEM_DATA_ALLOC) {
        std::cout << "Allocate ptr: " << std::hex << callback_data->data.mem_data.ptr
                  << std::dec << std::endl;
      } else {
        std::cout << "Free ptr: " << std::hex << callback_data->data.mem_data.ptr
                  << std::dec << std::endl;
      }
      if (callback_data->data.mem_data.device_type == TORCH_MONITOR_DEVICE_TYPE_CPU) {
        std::cout << "Device: CPU" << std::endl;
      } else if (callback_data->data.mem_data.device_type ==
                 TORCH_MONITOR_DEVICE_TYPE_GPU) {
        std::cout << "Device: GPU" << std::endl;
      } else {
        std::cout << "Device: Other" << std::endl;
      }
      std::cout << "Size: " << callback_data->data.mem_data.size << std::endl;
      std::cout << "Total size: " << callback_data->data.mem_data.total_allocated
                << std::endl;
      std::cout << "Total reserved: " << callback_data->data.mem_data.total_reserved
                << std::endl;
    }
  } else if (callback_site == TORCH_MONITOR_CALLBACK_EXIT) {
    if (callback_data->domain != TORCH_MONITOR_DOMAIN_MEMORY) {
      if (timestamp_enable) {
        std::cout << "Exit level: " << callback_data->data.op_data.nested_level << " at "
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count()
                  << std::endl;
      }
    }
    std::cout << "Exit Domain: " << callback_data->domain << std::endl;
    std::cout << "Current thread id: " << callback_data->current_thread_id << std::endl;
    std::cout << "Forward thread id: " << callback_data->data.op_data.forward_thread_id
              << std::endl;
    std::cout << "Sequence number: " << callback_data->data.op_data.sequence_number
              << std::endl;
    std::cout << "Name: " << std::string(callback_data->data.op_data.name) << std::endl;
    if (torch_monitor_outputs_capture_enable_get()) {
      callback_inputs_outputs_report(callback_site, callback_data);
    }
  }
}

void driver_env_init() {
  if (const char* env = std::getenv("TORCH_MONITOR_PYTHON_STATE_ENABLE")) {
    if (std::atoi(env) == 1) {
      python_state_enable = true;
    }
  }

  if (const char* env = std::getenv("TORCH_MONITOR_DRIVER_DEBUG")) {
    if (std::atoi(env) == 1) {
      driver_debug = true;
    }
  }

  if (const char* env = std::getenv("TORCH_MONITOR_TIMESTAMP_ENABLE")) {
    if (std::atoi(env) == 1) {
      timestamp_enable = true;
    }
  }

  if (const char* env = std::getenv("TORCH_MONITOR_VERBOSE_DISABLE")) {
    if (std::atoi(env) == 1) {
      verbose = false;
    }
  }

  if (const char* env = std::getenv("TORCH_MONITOR_INPUTS_CAPTURE")) {
    if (std::atoi(env) == 1) {
      torch_monitor_inputs_capture_enable_set(true);
    }
  }

  if (const char* env = std::getenv("TORCH_MONITOR_OUTPUTS_CAPTURE")) {
    if (std::atoi(env) == 1) {
      torch_monitor_outputs_capture_enable_set(true);
    }
  }
}

int driver_register() {
  driver_env_init();

  TORCH_MONITOR_CALL(torch_monitor_domain_enable, (TORCH_MONITOR_DOMAIN_FUNCTION));
  TORCH_MONITOR_CALL(torch_monitor_domain_enable,
                     (TORCH_MONITOR_DOMAIN_BACKWARD_FUNCTION));
  TORCH_MONITOR_CALL(torch_monitor_domain_enable, (TORCH_MONITOR_DOMAIN_MEMORY));
  TORCH_MONITOR_CALL(torch_monitor_callback_subscribe, (driver_callback));
  TORCH_MONITOR_CALL(torch_monitor_init, ());
  return 0;
}

int _ret = driver_register();
