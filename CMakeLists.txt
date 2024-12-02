cmake_minimum_required(VERSION 3.24)
# Run the following command to configure and build this project in a subfolder 'build'
#   cmake -B build -S . && cmake --build build/

if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.29)
    cmake_policy(SET CMP0156 NEW)
endif()
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.31)
    cmake_policy(SET CMP0179 NEW)
endif()


project(chub_block_solver)


find_package(MPI REQUIRED) # Look for an MPI library

# build solver from source files and link to libraries
add_executable(solver)
target_compile_features(solver PRIVATE cxx_std_17)
set_target_properties(solver PROPERTIES
    COMPILE_WARNING_AS_ERROR ON
    )
target_sources(solver
    PRIVATE
        main.cpp
    )
target_link_libraries(solver
    PUBLIC
        MPI::MPI_CXX
    )
target_compile_options(solver PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Wshadow>
    )
