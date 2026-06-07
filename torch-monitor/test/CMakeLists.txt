cmake_minimum_required(VERSION 3.12)

# Define the source and destination directories
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(DEST_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Create a custom target that will always be built
add_custom_target(copy_python_files ALL)

# Find all .py files in the source directory
file(GLOB PY_FILES ${SOURCE_DIR}/*.py)

# For each .py file, copy it to the destination directory
foreach(PY_FILE ${PY_FILES})
    add_custom_command(
        TARGET copy_python_files
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${PY_FILE}
                ${DEST_DIR}
    )
endforeach()

# Add unit tests of op and mem information
add_test(NAME add_cpu COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/add.py cpu)
set_tests_properties(add_cpu PROPERTIES
  ENVIRONMENT "LD_PRELOAD=${PROJECT_BINARY_DIR}/lib/driver.so")
add_test(NAME add_cuda COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/add.py cuda)
set_tests_properties(add_cuda PROPERTIES
  ENVIRONMENT "LD_PRELOAD=${PROJECT_BINARY_DIR}/lib/driver.so")

# Add tests training of a basic neural network
add_test(NAME mnist_cpu COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/mnist.py --epochs 1 --no-cuda --dry-run)
set_tests_properties(mnist_cpu PROPERTIES
  ENVIRONMENT "LD_PRELOAD=${PROJECT_BINARY_DIR}/lib/driver.so")
add_test(NAME mnist_cuda COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/mnist.py --epochs 1 --dry-run)
set_tests_properties(mnist_cuda PROPERTIES
  ENVIRONMENT "LD_PRELOAD=${PROJECT_BINARY_DIR}/lib/driver.so")

# Add tests inference of a large neural network
add_test(NAME resnet_cpu COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/resnet.py cpu)
set_tests_properties(resnet_cpu PROPERTIES
  ENVIRONMENT "LD_PRELOAD=${PROJECT_BINARY_DIR}/lib/driver.so")
add_test(NAME resnet_cuda COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/resnet.py cuda)
set_tests_properties(resnet_cuda PROPERTIES
  ENVIRONMENT "LD_PRELOAD=${PROJECT_BINARY_DIR}/lib/driver.so")

# Add tests capturing function inputs and outputs tensor data
add_test(NAME callback_io_cpu COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/callback_io.py cpu)
set_tests_properties(callback_io_cpu PROPERTIES
  ENVIRONMENT "LD_PRELOAD=${PROJECT_BINARY_DIR}/lib/driver.so")
add_test(NAME callback_io_cuda COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/callback_io.py cuda)
set_tests_properties(callback_io_cpu PROPERTIES
  ENVIRONMENT "LD_PRELOAD=${PROJECT_BINARY_DIR}/lib/driver.so")
