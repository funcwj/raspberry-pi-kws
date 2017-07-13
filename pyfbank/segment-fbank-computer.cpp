//
// Created by wujian on 17-4-23.
//

#include "segment-fbank-computer.h"

SegmentFbankComputer::SegmentFbankComputer(std::string config,
                                           std::string cmvn) {
    FbankOptions fbank_opts;
    ReadFeatsConfig(config, fbank_opts);
    fbank_computer_ = new Fbank(fbank_opts);
    ReadCmvnMatrix(cmvn);
}

void SegmentFbankComputer::ReadCmvnMatrix(std::string cmvn) {
    bool binary;
    Input ki(cmvn, &binary);
    cmvn_stats_.Read(ki.Stream(), binary);
}

void SegmentFbankComputer::ReadFeatsConfig(std::string config, FbankOptions &fbank_opts) {
    ParseOptions po("Parse for construction of the SegmentFbankComputer");
    fbank_opts.Register(&po);
    po.ReadConfigFile(config);
}

void SegmentFbankComputer::Compute(const VectorBase<BaseFloat> &waveform,
                                   Matrix<BaseFloat> *fbank_feature) {
    fbank_computer_->Compute(waveform, 1.0f, fbank_feature);
    ApplyCmvn(cmvn_stats_, true, fbank_feature);
}

np::ndarray SegmentFbankComputer::ComputeFbank(const np::ndarray &buffer) {

    KALDI_ASSERT(buffer.get_nd() == 1);
    KALDI_ASSERT(buffer.get_dtype() == np::dtype::get_builtin<float>());

    SubVector<BaseFloat> waveform(reinterpret_cast<BaseFloat*>(buffer.get_data()), buffer.shape(0));
    Compute(waveform, &fbank_feats_);
    return np::from_data(fbank_feats_.Data(), np::dtype::get_builtin<float>(),
                        py::make_tuple(fbank_feats_.NumRows(), fbank_feats_.NumCols()),
                         py::make_tuple(fbank_feats_.Stride() * sizeof(BaseFloat), sizeof(BaseFloat)),
                        py::object());
}