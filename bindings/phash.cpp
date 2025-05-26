#include <pybind11/pybind11.h>
#include "spatial_hash.hpp"

namespace py = pybind11;

PYBIND11_MODULE(phash, m)
{
    m.doc() = "Spatial hashing demo module";

    m.def("fill_dummy", [](){
        // 仅用于 smoke test
        return "phash imported OK";
    });
}
