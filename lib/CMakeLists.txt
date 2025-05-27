# lib/CMakeLists.txt
add_library(hash_cpu STATIC ${PROJECT_SOURCE_DIR}/src/cpu/hash.cpp)
target_include_directories(hash_cpu PUBLIC ${PROJECT_SOURCE_DIR}/include)

find_package(glm CONFIG REQUIRED)
add_library(spatial_hasher STATIC ${PROJECT_SOURCE_DIR}/src/spatial_hasher.cu)
target_link_libraries(spatial_hasher PRIVATE glm::glm-header-only)
target_include_directories(spatial_hasher PUBLIC ${PROJECT_SOURCE_DIR}/include)


# omp / cuda
find_package(OpenMP REQUIRED)
option(USE_OMP "Build OpenMP backend" ON)
if(USE_OMP AND OpenMP_CXX_FOUND)
    add_library(hash_omp STATIC ${PROJECT_SOURCE_DIR}/src/omp/hash_omp.cpp)
    target_link_libraries(hash_omp PUBLIC OpenMP::OpenMP_CXX)
    target_include_directories(hash_omp PUBLIC ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(hash_omp PRIVATE Thrust::Thrust)

endif()

option(USE_CUDA "Build CUDA backend" OFF)
if(USE_CUDA)
    enable_language(CUDA)
    add_library(hash_cuda STATIC ${PROJECT_SOURCE_DIR}/src/cuda/hash.cu)
    set_target_properties(hash_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(hash_cuda PUBLIC ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(hash_cuda PRIVATE Thrust::Thrust)

endif()