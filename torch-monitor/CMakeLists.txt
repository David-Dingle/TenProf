cmake_minimum_required(VERSION 3.12)

project(torch_monitor)

# The version number
set(TORCH_MONITOR_VERSION_MAJOR 1)
set(TORCH_MONITOR_VERSION_MINOR 0)
set(TORCH_MONITOR_VERSION_PATCH 0)

# Set the build standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the build type
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  if(DEBUG)
    set(CMAKE_BUILD_TYPE "Debug")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
  else()
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "-O3 -g -DNDEBUG")
  endif()
endif()

# # Option to control use of CXX11 ABI
# option(USE_CXX11_ABI "Use CXX11 ABI" OFF)

# # Conditionally add the flag for old ABI
# if(USE_CXX11_ABI)
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX11_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
# else()
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX11_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
# endif()

# Set the build output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/lib)

# Enable testing
enable_testing()

# Check set the torch install path
set(TORCH_DIR "" CACHE STRING "PyTorch install path")
if("${TORCH_DIR}" STREQUAL "")
  message(SEND_ERROR "TORCH_DIR must be specified")
endif()

# Check Python and set the compiler flags
find_package(Python 3.11.9 COMPONENTS Interpreter Development)
if(${Python_FOUND})
  message(STATUS "PYTHON detected: " ${Python_VERSION})

  # Check installed torch
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch"
    RESULT_VARIABLE TORCH_NOT_FOUND
  )

  if(NOT TORCH_NOT_FOUND)
    message(STATUS "torch is installed!")
  else()
    message(FATAL_ERROR "torch is NOT installed!")
  endif()

  # Check installed torchvision
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torchvision"
    RESULT_VARIABLE TORCHVISION_NOT_FOUND
  )

  if(NOT TORCHVISION_NOT_FOUND)
    message(STATUS "torchvision is installed!")
  else()
    message(FATAL_ERROR "torchvision is NOT installed!")
  endif()

  # Check installed torch CXX_ABI
  execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import torch; print(int(torch.compiled_with_cxx11_abi()))"
    OUTPUT_VARIABLE TORCH_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if (TORCH_CXX11_ABI STREQUAL "1")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=1")
    message("CXX11_ABI is used.")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    message("CXX11_ABI is not used.")
  endif()

else()
  message(SEND_ERROR "PYTHON not found")
endif()

set(Python_VERSION_MAJOR 3)
set(Python_VERSION_MINOR 11)
set(Python_LIBRARY_DIRS ${TORCH_DIR}/../../..)
set(Python_INCLUDE_DIRS ${TORCH_DIR}/../../../../include/python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR})

# Add the include directories
include_directories(
  ${TORCH_DIR}/include
  ${TORCH_DIR}/include/torch/csrc/api/include
  ${Python_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
)

# Add the link directories
link_directories(
  ${TORCH_DIR}/lib
  ${Python_LIBRARY_DIRS}
)

# Specify the sources
file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/src/*.cc")

# Define the library
add_library(${PROJECT_NAME} SHARED ${SOURCES})

# Link the libraries
target_link_libraries(${PROJECT_NAME}
  c10
  torch
  torch_cpu
  python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}
)

# Set rpath
set_target_properties(${PROJECT_NAME} PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "${TORCH_DIR}/lib;${Python_LIBRARY_DIRS}"
)

# Configure the header file
configure_file(
  ${PROJECT_SOURCE_DIR}/include/${PROJECT_NAME}.h
  ${PROJECT_BINARY_DIR}/include/${PROJECT_NAME}.h
)

# Set the library install destination
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

# Set the header install destination
install(FILES ${PROJECT_BINARY_DIR}/include/${PROJECT_NAME}.h DESTINATION include)

# Add subdirectories
add_subdirectory(driver)
add_subdirectory(test)
