cmake_minimum_required(VERSION 3.22)

set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9")
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/nvcc.exe")
set(CMAKE_CUDA_ARCHITECTURES 89)

project(MCFluid LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(Python3_EXECUTABLE  "C:/Users/Gargantua/anaconda3/envs/cuda_mcp/python.exe")
set(pybind11_DIR        "C:/Users/Gargantua/anaconda3/envs/cuda_mcp/Library/share/cmake/pybind11")

# -------- Subdirectories --------
add_subdirectory(lib)        #
add_subdirectory(bindings)   #

add_executable(test_spatial_hashing apps/test_spatial_hashing.cu)
target_include_directories(test_spatial_hashing PRIVATE include)
target_link_libraries(test_spatial_hashing PRIVATE hash_cpu)

find_package(Thrust REQUIRED CONFIG)
target_link_libraries(test_spatial_hashing PRIVATE Thrust::Thrust)

add_executable(vortex_method_mc apps/vortex_method_mc.cu)
find_package(glm CONFIG REQUIRED)
target_include_directories(vortex_method_mc PRIVATE include)
target_link_libraries(vortex_method_mc PRIVATE spatial_hasher)
target_link_libraries(vortex_method_mc PRIVATE glm::glm-header-only)
target_link_libraries(vortex_method_mc PRIVATE Thrust::Thrust)
target_link_libraries(spatial_hasher PRIVATE Thrust::Thrust)

if(USE_OMP)
    target_link_libraries(test_spatial_hashing PRIVATE hash_omp)
endif()
if(USE_CUDA)
    enable_language(CUDA)
    target_link_libraries(test_spatial_hashing PRIVATE hash_cuda)
    target_compile_definitions(test_spatial_hashing PRIVATE USE_CUDA_BACKEND)
    set_target_properties(test_spatial_hashing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


endif()

