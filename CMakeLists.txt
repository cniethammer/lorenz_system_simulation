cmake_minimum_required(VERSION 3.24)
# Run the following command to configure and build this project in a subfolder 'build'
#   cmake -B build -S . && cmake --build build/


project(chub_block_solver)


find_package(MPI REQUIRED) # Look for an MPI library

# build solver from source files and link to libraries
add_executable(solver)
target_sources(solver
    PRIVATE
        main.cpp
    )
target_link_libraries(solver
    PUBLIC
        MPI::MPI_CXX
    )
