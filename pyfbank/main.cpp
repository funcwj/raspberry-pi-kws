#include "segment-fbank-computer.h"

//using namespace kaldi;

BOOST_PYTHON_MODULE(pyfbank) {
    using namespace boost::python;
    np::initialize();
    class_<SegmentFbankComputer>("fbankcomputer", init<optional<std::string, std::string> >())
            .def("compute", &SegmentFbankComputer::ComputeFbank);
}


