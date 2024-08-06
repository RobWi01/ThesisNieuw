cmake_minimum_required(VERSION 3.4...3.18)
project(fastMultModule)

# Including pybind11 from the project directory
add_subdirectory(pybind11)

# Adding the Python module. Ensure 'FastMultWrapper.cpp' is correctly located in your project structure
pybind11_add_module(fastMultModule FastMultWrapper.cpp)

# Specify any required C++ standard here, if necessary
set_target_properties(fastMultModule PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
)

# Link any dependencies or include directories if needed
# target_link_libraries(fastMultModule PRIVATE some_library)
# target_include_directories(fastMultModule PRIVATE some/include/path)
