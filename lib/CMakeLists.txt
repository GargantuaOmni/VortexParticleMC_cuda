# lib/CMakeLists.txt
add_library(hash_cpu STATIC ${PROJECT_SOURCE_DIR}/src/cpu/hash.cpp)
target_include_directories(hash_cpu PUBLIC ${PROJECT_SOURCE_DIR}/include)

# find_package(glm CONFIG REQUIRED)
# add_library(spatial_hasher STATIC ${PROJECT_SOURCE_DIR}/src/spatial_hasher.cu)
# target_link_libraries(spatial_hasher PRIVATE glm::glm-header-only)
# target_include_directories(spatial_hasher PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_library(build_particle_set_cpu STATIC ${PROJECT_SOURCE_DIR}/src/cpu/build_particle_sets.cpp)
target_include_directories(build_particle_set_cpu PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_library(evaluate_vorticity_cpu STATIC ${PROJECT_SOURCE_DIR}/src/cpu/evaluate_vorticity.cpp)
target_include_directories(evaluate_vorticity_cpu PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_library(stb_write STATIC ${PROJECT_SOURCE_DIR}/src/others/stb_write.cpp)
target_include_directories(stb_write PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_library(monte_carlo_sampling STATIC ${PROJECT_SOURCE_DIR}/src/cpu/monte_carlo_sampling.cpp)
target_include_directories(monte_carlo_sampling PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_library(monte_carlo_bs STATIC ${PROJECT_SOURCE_DIR}/src/cpu/monte_carlo_bs.cpp)
target_include_directories(monte_carlo_bs PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(monte_carlo_bs PRIVATE monte_carlo_sampling)

add_library(simulation_step_cpu STATIC ${PROJECT_SOURCE_DIR}/src/cpu/simulation_step.cpp)
target_include_directories(simulation_step_cpu PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(simulation_step_cpu PRIVATE monte_carlo_sampling)
target_link_libraries(simulation_step_cpu PRIVATE monte_carlo_bs)

# omp / cuda
find_package(OpenMP REQUIRED)
option(USE_OMP "Build OpenMP backend" ON)
if(USE_OMP AND OpenMP_CXX_FOUND)
    add_library(hash_omp STATIC ${PROJECT_SOURCE_DIR}/src/omp/hash_omp.cpp)
    target_link_libraries(hash_omp PUBLIC OpenMP::OpenMP_CXX)
    target_include_directories(hash_omp PUBLIC ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(hash_omp PRIVATE Thrust::Thrust)


endif()

add_library(utilities STATIC ${PROJECT_SOURCE_DIR}/src/utilities.cu)
target_include_directories(utilities PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_options(utilities PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
)
target_compile_definitions(utilities PRIVATE USE_CUDA)

option(USE_CUDA "Build CUDA backend" OFF)
if(USE_CUDA)
    enable_language(CUDA)
    add_library(hash_cuda STATIC ${PROJECT_SOURCE_DIR}/src/cuda/hash.cu)
    set(CMAKE_CUDA_STANDARD 20)
    set_target_properties(hash_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(hash_cuda PUBLIC ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(hash_cuda PRIVATE Thrust::Thrust)
    target_compile_options(hash_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
    )

    add_library(build_particle_set_cuda STATIC ${PROJECT_SOURCE_DIR}/src/cuda/build_particle_sets.cu)
    target_include_directories(build_particle_set_cuda PUBLIC ${PROJECT_SOURCE_DIR}/include)
    target_compile_options(build_particle_set_cuda PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --expt-extended-lambda>
    )
    target_compile_definitions(build_particle_set_cuda PUBLIC USE_CUDA)
endif()