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
find_package(yaml-cpp) # look for yaml-cpp
if(NOT ${yaml-cpp_FOUND}) #  install yaml-cpp from source if not found on the system
    include(FetchContent)
    FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG 0.8.0 # Can be a tag (yaml-cpp-x.x.x), a commit hash, or a branch name (master)
    )
    FetchContent_MakeAvailable(yaml-cpp)
endif()

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
        yaml-cpp::yaml-cpp
    )
target_compile_options(solver PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
    $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic -Wshadow>
    )
