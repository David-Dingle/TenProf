#ifndef TORCH_MONITOR_H
#define TORCH_MONITOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
#  define EXTERNC extern "C"
#else
#  define EXTERNC
#endif

#define TORCH_MONITOR_MAX_INPUTS_OUTPUTS_SIZE 128
#define TORCH_MONITOR_MAX_TENSOR_DIMENSION 128

/**
 * @brief Error code
 *
 */
typedef enum torch_monitor_status {
  TORCH_MONITOR_STATUS_SUCCESS = 0,
  TORCH_MONITOR_STATUS_INIT_HANDLE_FAIL = 1,
  TORCH_MONITOR_STATUS_INIT_DOMAIN_NULL = 2,
  TORCH_MONITOR_STATUS_INIT_MEMORY_NOT_REGISTER = 3,
  TORCH_MONITOR_STATUS_ENABLE_DOMAIN_OUT_RANGE = 4,
  TORCH_MONITOR_STATUS_SUBSCRIBE_EXIST = 5,
  TORCH_MONITOR_STATUS_SUBSCRIBE_SUBSCRIBER_NULL = 6,
  TORCH_MONITOR_STATUS_FINALIZE_NOT_INIT = 7,
  TORCH_MONITOR_STATUS_FINALIZE_MEMORY_FAIL = 8,
  TORCH_MONITOR_STATUS_PYTHON_STATES_NULL = 9,
  TORCH_MONITOR_STATUS_COUNT = 10
} torch_monitor_status_t;

/**
 * @brief Monitor domains. One or more domains can be enabled in a single run.
 *
 */
typedef enum torch_monitor_domain {
  // c10/ATen ops, autograd nodes
  TORCH_MONITOR_DOMAIN_FUNCTION = 0,
  // Functions/nodes called from the autograd
  TORCH_MONITOR_DOMAIN_BACKWARD_FUNCTION = 1,
  // TorchScript functions, methods
  TORCH_MONITOR_DOMAIN_TORCHSCRIPT_FUNCTION = 2,
  // Kernel Function dtype Tag
  TORCH_MONITOR_DOMAIN_KERNEL_FUNCTION_DTYPE = 3,
  // Torchbind custom class
  TORCH_MONITOR_DOMAIN_CUSTOM_CLASS = 4,
  // Generic Build Feature
  TORCH_MONITOR_DOMAIN_BUILD_FEATURE = 5,
  // Python line interpreter (i.e., could be removed)
  TORCH_MONITOR_DOMAIN_LITE_INTERPRETER = 6,
  // User defined scope (e.g. with record_function())
  TORCH_MONITOR_DOMAIN_USER_SCOPE = 7,
  // Scopes for static runtime, a specialized TorchScript interpreter
  TORCH_MONITOR_DOMAIN_STATIC_RUNTIME_OP = 8,
  TORCH_MONITOR_DOMAIN_STATIC_RUNTIME_MODEL = 9,
  // Memory allocation
  TORCH_MONITOR_DOMAIN_MEMORY = 10,
  TORCH_MONITOR_DOMAIN_COUNT = 11
} torch_monitor_domain_t;

/**
 * @brief Enter or exit a torch function
 *
 */
typedef enum torch_monitor_callback_site {
  TORCH_MONITOR_CALLBACK_ENTER = 0,
  TORCH_MONITOR_CALLBACK_EXIT = 1,
  TORCH_MONITOR_CALLBACK_COUNT = 2
} torch_monitor_callback_site_t;

/**
 * @brief The device type of a memory or operation
 *
 */
typedef enum torch_monitor_device_type {
  TORCH_MONITOR_DEVICE_TYPE_CPU = 0,
  TORCH_MONITOR_DEVICE_TYPE_GPU = 1,
  TORCH_MONITOR_DEVICE_TYPE_FPGA = 2,
  TORCH_MONITOR_DEVICE_TYPE_COUNT = 3
} torch_monitor_device_type_t;

/**
 * @brief The state of a PyTorch thread
 *
 */
typedef enum torch_monitor_thread_state {
  TORCH_MONITOR_THREAD_STATE_NONE = 0x0,
  TORCH_MONITOR_THREAD_STATE_FORWARD = 0x1,
  TORCH_MONITOR_THREAD_STATE_BACKWARD = (0x1 << 1),
  TORCH_MONITOR_THREAD_STATE_OPTIMIZER = (0x1 << 2),
  TORCH_MONITOR_THREAD_STATE_ACTIVE = (0x1 << 3),
  TORCH_MONITOR_THREAD_STATE_IDLE = (0x1 << 4),
  TORCH_MONITOR_THREAD_STATE_INVALID = (0x1 << 5),
} torch_monitor_thread_state_t;

/**
 * @brief Tensor dtype
 *
 * */
typedef enum torch_monitor_scalar_type {
  TORCH_MONITOR_SCALAR_TYPES_BYTE = 0,
  TORCH_MONITOR_SCALAR_TYPES_CHAR = 1,
  TORCH_MONITOR_SCALAR_TYPES_SHORT = 2,
  TORCH_MONITOR_SCALAR_TYPES_INT = 3,
  TORCH_MONITOR_SCALAR_TYPES_LONG = 4,
  TORCH_MONITOR_SCALAR_TYPES_HALF = 5,
  TORCH_MONITOR_SCALAR_TYPES_FLOAT = 6,
  TORCH_MONITOR_SCALAR_TYPES_DOUBLE = 7,
  TORCH_MONITOR_SCALAR_TYPES_COMPLEXHALF = 8,
  TORCH_MONITOR_SCALAR_TYPES_COMPLEXFLOAT = 9,
  TORCH_MONITOR_SCALAR_TYPES_COMPLEXDOUBLE = 10,
  TORCH_MONITOR_SCALAR_TYPES_BOOL = 11,
  TORCH_MONITOR_SCALAR_TYPES_QINT8 = 12,
  TORCH_MONITOR_SCALAR_TYPES_QUINT8 = 13,
  TORCH_MONITOR_SCALAR_TYPES_QINT32 = 14,
  TORCH_MONITOR_SCALAR_TYPES_BFLOAT16 = 15,
  TORCH_MONITOR_SCALAR_TYPES_QUINT4X2 = 16,
  TORCH_MONITOR_SCALAR_TYPES_QUINT2X4 = 17,
  TORCH_MONITOR_SCALAR_TYPES_BITS1X8 = 18,
  TORCH_MONITOR_SCALAR_TYPES_BITS2X4 = 19,
  TORCH_MONITOR_SCALAR_TYPES_BITS4X2 = 20,
  TORCH_MONITOR_SCALAR_TYPES_BITS8 = 21,
  TORCH_MONITOR_SCALAR_TYPES_BITS16 = 22,
  TORCH_MONITOR_SCALAR_TYPES_FLOAT8_E5M2 = 23,
  TORCH_MONITOR_SCALAR_TYPES_FLOAT8_E4M3FN = 24,
  TORCH_MONITOR_SCALAR_TYPES_FLOAT8_E5M2FNUZ = 25,
  TORCH_MONITOR_SCALAR_TYPES_FLOAT8_E4M3FNUZ = 26,
  TORCH_MONITOR_SCALAR_TYPES_UNMATCHED_TYPE = 27,
} torch_monitor_scalar_type_t;

/**
 * @brief Values are extracted from tensor basic info
 *
 * */
typedef struct torch_monitor_callback_tensor_data {
  int64_t index;
  int64_t numel;
  int64_t dim;
  torch_monitor_scalar_type_t dtype;
  uint64_t itemsize;
  int64_t storage_offset;
  int64_t sizes[TORCH_MONITOR_MAX_TENSOR_DIMENSION];
  int64_t strides[TORCH_MONITOR_MAX_TENSOR_DIMENSION];
  void* data_ptr;
  void* metadata_ptr;
} torch_monitor_callback_tensor_data_t;

/**
 * @brief A structure stores callback inputs or outputs
 *
 * */
typedef struct torch_monitor_input_output_data {
  // a size indicator of function inputs or function outputs
  int64_t size;
  // May need a Ivalue wrapper struct to support more data types
  torch_monitor_callback_tensor_data_t tensor_data[TORCH_MONITOR_MAX_INPUTS_OUTPUTS_SIZE];
} torch_monitor_input_output_data_t;

/**
 * @brief Information of each aten operation
 * The <forward_thread_id, sequence_number> pair records the
 * the last forward operation with specific forward thread id and sequence number.
 * We attribute a backward operation to the master frame of that operation.
 * There can be multiple <1, 0> pairs, the backward operation with seq=0 only
 * attributes to the most recent forward operation with seq=0.
 *
 */
typedef struct torch_monitor_op_data {
  uint64_t forward_thread_id;
  // sequence_number = -1 means this op does not have a backward counterpart
  int64_t sequence_number;
  // An aten op calls another aten op
  //               op1->op2->op3
  // nested_level: 0->1->2
  //               |
  //             master
  uint32_t nested_level;
  const char* name;
  torch_monitor_input_output_data_t input_output_data;
} torch_monitor_op_data_t;

/**
 * @brief Memory allocation or free
 *
 */
typedef enum torch_monitor_mem_data_type {
  TORCH_MONITOR_MEM_DATA_ALLOC = 0,
  TORCH_MONITOR_MEM_DATA_FREE = 1,
  TORCH_MONITOR_MEM_DATA_COUNT = 2
} torch_monitor_mem_data_type_t;

/**
 * @brief Information of each torch memory alloc operation
 *
 */
typedef struct torch_monitor_mem_data {
  torch_monitor_mem_data_type_t type;
  torch_monitor_device_type_t device_type;
  void* ptr;
  int64_t size;
  int64_t total_allocated;
  int64_t total_reserved;
} torch_monitor_mem_data_t;

/**
 * @brief Information about each python frame
 *
 */
typedef struct torch_monitor_python_state {
  const char* file_name;
  const char* function_name;
  size_t function_first_lineno;
  size_t lineno;
} torch_monitor_python_state_t;

/**
 * @brief General callback information container
 *
 */
typedef struct torch_monitor_callback_data {
  torch_monitor_domain_t domain;
  uint64_t current_thread_id;

  // data can be casted using domain to
  // torch_monitor_callback_op_data_t
  // torch_monitor_callback_mem_data_t
  union {
    torch_monitor_op_data_t op_data;
    torch_monitor_mem_data_t mem_data;
  } data;
} torch_monitor_callback_data_t;

/**
 * @brief A callback that handles callback_data at each pytorch function enter/exit
 *
 * @param callback_site
 * @param callback_data
 *
 */
typedef void (*torch_monitor_callback_func_t)(
    torch_monitor_callback_site_t callback_site,
    torch_monitor_callback_data_t* callback_data);

/**
 * @brief A callback that handles callback_data at each pytorch function enter/exit
 *
 * @param func The callback to register
 * @return torch_monitor_status_t
 *
 * @note not thread safe
 *
 */
EXTERNC torch_monitor_status_t
torch_monitor_callback_subscribe(torch_monitor_callback_func_t func);

/**
 * @brief Enable a domain to be monitored
 *
 * @param domain The domain to monitor
 * @return torch_monitor_status_t
 *
 * @note not thread safe
 *
 */
EXTERNC torch_monitor_status_t torch_monitor_domain_enable(torch_monitor_domain_t domain);

/**
 * @brief Query the python states of the query thread
 *
 * @param max_num_states Returns up to num_states frames
 * @param states An array of states allocated by the tool but not torch_monitor
 * @param num_states Number of states collected
 * @return torch_monitor_status_t
 *
 * @note: not thread safe
 *
 */
EXTERNC torch_monitor_status_t torch_monitor_python_state_get(
    size_t max_num_states, torch_monitor_python_state_t* states, size_t* num_states);

/**
 * @brief Start monitoring pytorch functions in registered domains.
 * This function should be called only once at process initialization.
 *
 * @return torch_monitor_status_t
 *
 * @note: not thread safe
 *
 */
EXTERNC torch_monitor_status_t torch_monitor_init();

/**
 * @brief Init thread local states. This function should be called when each thread
 * initializes.
 *
 * @return torch_monitor_status_t
 *
 * @note not thread safe
 *
 */
EXTERNC torch_monitor_status_t torch_monitor_thread_init();

/**
 * @brief Unregister all callbacks. This function should be called only once at process
 * termination.
 *
 * @return torch_monitor_status_t
 *
 * @note not thread safe
 *
 */
EXTERNC torch_monitor_status_t torch_monitor_finalize();

/**
 * @brief Clear thread local states. This function should be called when each thread
 * terminates.
 *
 * @return torch_monitor_status_t
 *
 * @note not thread safe
 *
 */
EXTERNC torch_monitor_status_t torch_monitor_thread_finalize();

/**
 * @brief Access functions for the environmental variables
 *
 */
EXTERNC bool torch_monitor_inputs_capture_enable_get();

EXTERNC void torch_monitor_inputs_capture_enable_set(bool val);

EXTERNC bool torch_monitor_outputs_capture_enable_get();

EXTERNC void torch_monitor_outputs_capture_enable_set(bool val);

/**
 * @brief A getter functions for tensor dtype name
 *
 */
EXTERNC const char* torch_monitor_dtype_name_get(torch_monitor_scalar_type_t scalar_type);

#endif  // TORCH_MONITOR_H
