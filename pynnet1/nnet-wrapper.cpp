#include "nnet-wrapper.h"


BaseFloat Cos(CuVectorBase<BaseFloat> &vec1, CuVectorBase<BaseFloat> &vec2) {
    BaseFloat dot_product = VecVec(vec1, vec2);
    BaseFloat mod_product = vec1.Norm(2) * vec2.Norm(2);
    return dot_product / mod_product;
}

BaseFloat KL(CuVectorBase<BaseFloat> &vec1, CuVectorBase<BaseFloat> &vec2) {
    BaseFloat dis = 0.0;
    for(int i = 0; i < vec1.Dim(); i++) {
        dis += vec2(i) * Log(vec2(i) / vec1(i));
    }
    return dis;
}

void EvalDist(CuMatrixBase<BaseFloat> &mat, CuVectorBase<BaseFloat> &vec, CuVector<BaseFloat> *dis) {
    KALDI_ASSERT(mat.NumCols() == vec.Dim());
    dis->Resize(mat.NumRows());
    for (int i = 0; i < dis->Dim(); i++) {
        CuSubVector<BaseFloat> sub(mat, i);
        (*dis)(i) = -Log(Cos(sub, vec));
    }
}


int32 Next(int32 v, int32 round) {
    return (v + 1) % round;
}

int32 ArgminTriple(BaseFloat *triple, int32 *idx) {
    BaseFloat min_val = triple[0];
    if (idx != NULL)
        *idx = 0;
    for (int i = 1; i < 3; i++) {
        if (min_val > triple[i]) {
            min_val = triple[i];
            *idx = i;
        }
    }
    return min_val;
}



NnetWrapper::NnetWrapper(std::string nnet_mdl, std::string vec_in) {
    nnet_.Read(nnet_mdl);
    KALDI_ASSERT(ClassifyRspecifier("ark:" + vec_in, NULL, NULL) != kNoRspecifier);
    SequentialBaseFloatMatrixReader read("ark:" + vec_in);
    template_ = read.Value();
    KALDI_LOG << "length of the template: " << template_.NumRows();
    segment_ = new CuMatrix<BaseFloat>(template_.NumRows(), template_.NumRows());
    accumulate_cost_ = new CuMatrix<BaseFloat>(2, template_.NumRows());
    accumulate_step_ = new CuMatrix<BaseFloat>(2, template_.NumRows());
    start_base_ = initilized_ = cur_state_ = 0;
}

NnetWrapper::NnetWrapper(std::string nnet_mdl) {
    nnet_.Read(nnet_mdl);
}

bool NnetWrapper::IsSpotting(int32 eval_method) {
    BaseFloat sum = 0.0;
    for (float score: score_buffer_) {
        switch(eval_method){
        case 1:
            sum += score;
            break;
        case 2:
            if(score >= spot_threshold_)
                return false;
            break;
        default:
            KALDI_ERR << "unknown evaluate method[support 1/2 only]";
            break;
        }
    }
    return eval_method == 1 ? sum / score_buffer_.size() <= spot_threshold_: true;
}

void NnetWrapper::SetDebug(bool debug) {
    debug_ = debug;
}

void NnetWrapper::SetSpotThreshold(BaseFloat thres) {
    spot_threshold_ = thres;
}

void NnetWrapper::SetWindowSize(int32 wnd_size) {
    window_size_ = wnd_size;
}

void NnetWrapper::FeedForward(np::ndarray &vector) {

    KALDI_ASSERT(vector.get_dtype() == np::dtype::get_builtin<float>());
    KALDI_ASSERT(vector.get_nd() <= 2);

    int cols = vector.shape(vector.get_nd() - 1);
    KALDI_ASSERT(cols == nnet_.InputDim());

    int rows = vector.get_nd() == 1 ? 1: vector.shape(0);

    CuSubMatrix<BaseFloat> nnet_in(reinterpret_cast<BaseFloat*>(vector.get_data()),
                                   rows, cols, vector.strides(0) / sizeof(BaseFloat));
    nnet_.Feedforward(nnet_in, &nnet_out_);
}


BaseFloat NnetWrapper::ApplySegmentDTW() {
    int cur_state = 1, prev_state = 0, orig_base = start_base_, i = start_base_;

    // initilized from start_base_
    for(int j = 0; j < segment_->NumCols(); j++) {
        if (j == 0)
            (*accumulate_cost_)(prev_state, j) = (*segment_)(i, j);
        else
            (*accumulate_cost_)(prev_state, j) = (*segment_)(i, j) + (*accumulate_cost_)(prev_state, j - 1);
        (*accumulate_step_)(prev_state, j) = j + 1;
    }

    for (int round = 1; round < segment_->NumRows(); round++) {
        i = Next(i, segment_->NumRows());

        (*accumulate_step_)(cur_state, 0) = i + 1;
        (*accumulate_cost_)(cur_state, 0) = (*accumulate_cost_)(prev_state, 0) + (*segment_)(i, 0);
        for (int j = 1; j < segment_->NumRows(); j++) {
            BaseFloat cost_triple[3] = { (*accumulate_cost_)(cur_state, j - 1), (*accumulate_cost_)(prev_state, j - 1),
                    (*accumulate_cost_)(prev_state, j) };
            BaseFloat step_triple[3] = { (*accumulate_step_)(cur_state, j - 1), (*accumulate_step_)(prev_state, j - 1),
                    (*accumulate_step_)(prev_state, j) };
            int32 min_idx = 0;
            (*accumulate_cost_)(cur_state, j) = ArgminTriple(cost_triple, &min_idx) + (*segment_)(i, j);
            KALDI_ASSERT(min_idx >= 0 && min_idx < 3);
            (*accumulate_step_)(cur_state, j) = step_triple[min_idx] + 1;
        }
        // swap instead of data copy
        std::swap(cur_state, prev_state);
    }
    KALDI_ASSERT(Next(i, segment_->NumRows()) == orig_base);
    return (*accumulate_cost_)(prev_state, segment_->NumRows() - 1) / (*accumulate_step_)(prev_state, segment_->NumRows() - 1);
}

void NnetWrapper::UpdateStartBase(int32 method_idx) {
    // set next position to replace
    // same as 'start_base_ = (start_base_ + 1) % segment_->NumRows()'
    start_base_ = Next(start_base_, segment_->NumRows());
    if (start_base_ == 0)
        initilized_ = 1;
    if (initilized_ == 1) {
        BaseFloat score = ApplySegmentDTW();
        if (debug_)
            std::cout << score << std::endl;
        score_buffer_.push_back(score);
        if (score_buffer_.size() == window_size_) {
            if (IsSpotting(method_idx) && cur_state_ == 0) {
                cur_state_ = 1;
                // std::cout << "Spotting!" << std::endl;
                KALDI_LOG << "Spotting!";
            } else if (!IsSpotting(method_idx)){
                cur_state_ = 0;
            }
            score_buffer_.erase(score_buffer_.begin());
        }
//        Timer timer;
//        std::cout << ApplySegmentDTW() << std::endl;
//        KALDI_LOG << "segment dtw cost " << timer.Elapsed() << " s";
    }
}

void NnetWrapper::PostProcess(np::ndarray &vector, int32 method_idx) {
    FeedForward(vector);
    KALDI_ASSERT(nnet_out_.NumRows() == 1);
    CuVector<BaseFloat> dis;

    CuSubVector<BaseFloat> nnet_out_vec(nnet_out_, 0);
    EvalDist(template_, nnet_out_vec, &dis);

    CuSubVector<BaseFloat> sub((*segment_), start_base_);
    sub.CopyFromVec(dis);

    UpdateStartBase(method_idx);
}

np::ndarray NnetWrapper::Predict(np::ndarray &vector) {

    int rows = vector.get_nd() == 1 ? 1: vector.shape(0);

    FeedForward(vector);
    // std::cout << nnet_out_.Row(0) << std::endl;

    return np::from_data(nnet_out_.Data(), np::dtype::get_builtin<float>(),
                         py::make_tuple(rows, nnet_out_.NumCols()),
                         py::make_tuple(nnet_out_.Stride() * sizeof(BaseFloat), sizeof(BaseFloat)),
                         py::object());
}
