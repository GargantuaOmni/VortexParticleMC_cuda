set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

pybind11_add_module(phash phash.cpp)        #
target_include_directories(phash PRIVATE ${PROJECT_SOURCE_DIR}/include)
# target_link_libraries(phash PRIVATE hash_cpu)

add_custom_command(
        TARGET phash
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E echo ">>> Copying phash to project root"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:phash>          #
        ${PROJECT_SOURCE_DIR}
        COMMENT "Copy phash.pyd to project root"
)