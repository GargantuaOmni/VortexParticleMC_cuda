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
# add_subdirectory(bindings)   #

add_executable(test_spatial_hashing apps/test_spatial_hashing.cu)
target_include_directories(test_spatial_hashing PRIVATE include)
target_link_libraries(test_spatial_hashing PRIVATE hash_cpu)

if(USE_CUDA)
else()
include_directories(${PROJECT_SOURCE_DIR}/deps/cccl)
include_directories(${PROJECT_SOURCE_DIR}/deps/cccl/thrust)
include_directories(${PROJECT_SOURCE_DIR}/deps/cccl/cub)
include_directories(${PROJECT_SOURCE_DIR}/deps/cccl/libcudacxx/include)
endif()

find_package(Thrust REQUIRED CONFIG)
target_link_libraries(test_spatial_hashing PRIVATE Thrust::Thrust)

add_executable(vortex_method_mc src/vortex_particle_mc.cu)
find_package(glm CONFIG REQUIRED)
target_include_directories(vortex_method_mc PRIVATE include)
# target_link_libraries(vortex_method_mc PRIVATE spatial_hasher)
target_link_libraries(vortex_method_mc PRIVATE glm::glm-header-only)
target_link_libraries(vortex_method_mc PRIVATE build_particle_set_cpu)
target_link_libraries(vortex_method_mc PRIVATE evaluate_vorticity_cpu)
target_link_libraries(vortex_method_mc PRIVATE monte_carlo_sampling)
target_link_libraries(vortex_method_mc PRIVATE monte_carlo_bs)
target_link_libraries(vortex_method_mc PRIVATE simulation_step_cpu)

target_link_libraries(vortex_method_mc PRIVATE hash_cpu)
target_link_libraries(vortex_method_mc PRIVATE stb_write)

if(USE_OMP)
    target_link_libraries(test_spatial_hashing PRIVATE hash_omp)
endif()
if(USE_CUDA)
    enable_language(CUDA)
    add_definitions(-DUSE_CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

    target_link_libraries(test_spatial_hashing PRIVATE hash_cuda)
    target_compile_definitions(test_spatial_hashing PRIVATE USE_CUDA_BACKEND)
    set_target_properties(test_spatial_hashing PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

    target_link_libraries(vortex_method_mc PRIVATE hash_cuda)
    target_link_libraries(vortex_method_mc PRIVATE build_particle_set_cuda)
    target_link_libraries(vortex_method_mc PRIVATE utilities)
    target_link_libraries(vortex_method_mc PRIVATE monte_carlo_bs_cuda)
    target_link_libraries(vortex_method_mc PRIVATE simulation_step_cuda)
    target_compile_definitions(vortex_method_mc PUBLIC USE_CUDA)

    set_target_properties(vortex_method_mc PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_STANDARD 20                #
            CUDA_STANDARD_REQUIRED ON
    )
    target_compile_options(vortex_method_mc PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
    )


else()
    target_link_libraries(vortex_method_mc PRIVATE utilities)

endif()

target_link_libraries(vortex_method_mc PRIVATE Thrust::Thrust)
# target_link_libraries(spatial_hasher PRIVATE Thrust::Thrust)