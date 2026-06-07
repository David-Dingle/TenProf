#include "utils.h"

#include "torch_monitor.h"

namespace torch_monitor {

torch_monitor_domain_t aten_scope_match(at::RecordScope scope) {
  switch (scope) {
    case at::RecordScope::FUNCTION:
      return TORCH_MONITOR_DOMAIN_FUNCTION;
    case at::RecordScope::BACKWARD_FUNCTION:
      return TORCH_MONITOR_DOMAIN_BACKWARD_FUNCTION;
    default:
      return TORCH_MONITOR_DOMAIN_COUNT;
  }
}

at::RecordScope torch_monitor_domain_match(torch_monitor_domain_t domain) {
  switch (domain) {
    case TORCH_MONITOR_DOMAIN_FUNCTION:
      return at::RecordScope::FUNCTION;
    case TORCH_MONITOR_DOMAIN_BACKWARD_FUNCTION:
      return at::RecordScope::BACKWARD_FUNCTION;
    default:
      return at::RecordScope::NUM_SCOPES;
  }
}

torch_monitor_device_type_t aten_device_type_match(at::DeviceType device_type) {
  switch (device_type) {
    case at::DeviceType::CPU:
      return TORCH_MONITOR_DEVICE_TYPE_CPU;
    case at::DeviceType::CUDA:
    case at::DeviceType::HIP:
      return TORCH_MONITOR_DEVICE_TYPE_GPU;
    case at::DeviceType::FPGA:
      return TORCH_MONITOR_DEVICE_TYPE_FPGA;
    default:
      return TORCH_MONITOR_DEVICE_TYPE_COUNT;
  }
}

}  // namespace torch_monitor
