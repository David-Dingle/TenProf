cmake_minimum_required(VERSION 3.12)

# Add the necessary flags, include, and library directories
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${PROJECT_BINARY_DIR}/lib)

# Specify the sources and output shared library
set(SOURCES driver.cc)
add_library(driver SHARED ${SOURCES})
target_link_libraries(driver ${PROJECT_NAME})

# Set the shared library name to be the same as the target name
set_target_properties(driver PROPERTIES PREFIX "")

# Set the rpath so the shared library knows where to find libtorch_monitor
set_target_properties(driver PROPERTIES BUILD_RPATH ${PROJECT_BINARY_DIR}/lib)
set_target_properties(driver PROPERTIES INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

# Set the install destination
install(TARGETS driver DESTINATION lib)
