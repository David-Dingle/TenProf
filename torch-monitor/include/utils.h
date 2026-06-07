#ifndef TORCH_MONITOR_UTILS_H
#define TORCH_MONITOR_UTILS_H

#include <torch/extension.h>

#include <cstdio>

#include "torch_monitor.h"

#ifdef DEBUG
#  define LOG_INFO(...)                    \
    do {                                   \
      fprintf(stdout, "TORCH_MONITOR-> "); \
      fprintf(stdout, __VA_ARGS__);        \
      fprintf(stdout, "\n");               \
    } while (0)
#else
#  define LOG_INFO(...)
#endif

namespace torch_monitor {

torch_monitor_domain_t aten_scope_match(at::RecordScope scope);

at::RecordScope torch_monitor_domain_match(torch_monitor_domain_t domain);

torch_monitor_device_type_t aten_device_type_match(at::DeviceType device_type);

}  // namespace torch_monitor

#endif  // TORCH_MONITOR_UTILS_H
