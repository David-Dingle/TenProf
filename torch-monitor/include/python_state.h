#ifndef TORCH_MONITOR_PYTHON_STATE_H
#define TORCH_MONITOR_PYTHON_STATE_H

#include <Python.h>

#include <string>
#include <vector>

#include "torch_monitor.h"

namespace torch_monitor {

struct PythonState {
  std::string file_name;
  std::string function_name;
  size_t function_first_lineno;
  size_t lineno;

  PythonState(const std::string& file_name, const std::string& function_name,
              size_t function_first_lineno, size_t lineno)
      : file_name(file_name),
        function_name(function_name),
        function_first_lineno(function_first_lineno),
        lineno(lineno) {
  }
};

class PythonStateMonitor {
 public:
  // Return the current python states with a query or using the previous cached states
  std::vector<PythonState>& get_states(bool cached = false);

  // Get the singleton instance
  static PythonStateMonitor& instance();

 private:
  PythonStateMonitor() {
  }

  std::string unpack_pyobject(PyObject* obj);

 private:
  // Cached states for each thread
  static inline thread_local std::vector<PythonState> _states;
};

}  // namespace torch_monitor

#endif  // TORCH_MONITOR_PYTHON_STATE_H
