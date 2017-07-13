//
// Created by wujian on 17-4-23.
//

#ifndef SEGMENT_FBANK_COMPUTER_H
#define SEGMENT_FBANK_COMPUTER_H

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "feat/feature-fbank.h"
#include "transform/cmvn.h"

using namespace kaldi;

namespace py = boost::python;
namespace np = py::numpy;

class SegmentFbankComputer {

public:
    SegmentFbankComputer(std::string config = "fbank.conf",
                         std::string cmvn = "cmvn.global");
    void Compute(const VectorBase<BaseFloat> &waveform,
                 Matrix<BaseFloat> *fbank_feature);
    void ReadFeatsConfig(std::string config, FbankOptions &fbank_opts);
    void ReadCmvnMatrix(std::string cmvn);
    np::ndarray ComputeFbank(const np::ndarray &buffer);
private:
    Fbank *fbank_computer_;
    Matrix<double> cmvn_stats_;
    Matrix<BaseFloat> fbank_feats_;
};


#endif //SEGMENT_FBANK_COMPUTER_H
