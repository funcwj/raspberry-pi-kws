//
// Created by wujian on 17-5-10.
//

#include "nnet-wrapper.h"


BOOST_PYTHON_MODULE(pynnet1) {
    using namespace boost::python;
    np::initialize();
    class_<NnetWrapper>("nnet1", init<std::string, optional<std::string> >())
    .def("predict", &NnetWrapper::Predict) \
        .def("postprocess", &NnetWrapper::PostProcess) \
    .def("debug", &NnetWrapper::SetDebug) \
        .def("threshold", &NnetWrapper::SetSpotThreshold);

}