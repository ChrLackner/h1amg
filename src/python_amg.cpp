
#include <python_ngstd.hpp>
#include <comp.hpp>
#include "h1amg.hpp"

using namespace ngcomp;
using namespace h1amg;

PYBIND11_MODULE(h1amg,m)
{
  py::module::import("ngsolve");

  auto h1amg = py::class_<H1AMG, shared_ptr<H1AMG>, Preconditioner> (m,"H1AMG");
  h1amg
    .def(py::init([h1amg](shared_ptr<BilinearForm> bfa, py::kwargs kwargs)
                  {
                    auto flags = CreateFlagsFromKwArgs(h1amg,kwargs);
                    return make_shared<H1AMG>(bfa,flags);
                  }), py::arg("Bilinearform"))
    ;
}
