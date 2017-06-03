//
// Created by wujian on 17-5-10.
//

#ifndef NNET_WRAPPER_H
#define NNET_WRAPPER_H


#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include <vector>
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-loss.h"
#include "util/common-utils.h"

using namespace kaldi;
using namespace kaldi::nnet1;

namespace py = boost::python;
namespace np = boost::python::numpy;

class NnetWrapper {

public:
    NnetWrapper(std::string nnet_mdl, std::string mat_in);
    NnetWrapper(std::string nnet_mdl);
    void FeedForward(np::ndarray &vector);
    void PostProcess(np::ndarray &vector, int32 method_idx);
    void UpdateStartBase(int32 method_idx);
    BaseFloat ApplySegmentDTW();
    bool IsSpotting(int32 eval_method);
    void SetDebug(bool debug);
    void SetSpotThreshold(BaseFloat thres);
    void SetWindowSize(int32 wnd_size);
    np::ndarray Predict(np::ndarray &vector);
private:
    // kaldi nnet object
    Nnet nnet_;
    // keep memory not free
    CuMatrix<BaseFloat> nnet_out_;
    CuMatrix<BaseFloat> template_;

    // full matrix, T X T
    CuMatrix<BaseFloat> *segment_;
    // 2 rows, save space
    CuMatrix<BaseFloat> *accumulate_cost_;
    CuMatrix<BaseFloat> *accumulate_step_;

    int32 start_base_ = 0;
    int32 initilized_ = 0;
    // scoring
    std::vector<BaseFloat> score_buffer_;
    int32 cur_state_;
    BaseFloat window_size_ = 10;
    BaseFloat spot_threshold_  = 0.4;

    // debug flag
    bool debug_ = true;
};

#endif //NNET_WRAPPER_H
