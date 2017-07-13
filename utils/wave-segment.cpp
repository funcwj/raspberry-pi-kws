#include <iostream>
#include "feat/wave-reader.h"
#include "util/common-utils.h"

using namespace kaldi;

class VAD {
    typedef std::vector<std::pair<int32, int32> > Segment;
    enum Status { kSilence, kActive };

public:
    VAD(BaseFloat threshold, BaseFloat window) {
        energy_threshold_ = threshold;
        disturb_frame_ = window;
        cur_status_ = kSilence;
        cur_step_ = start_frame_ = 0;
        cur_frame_ = -1;
    }

    void Run(BaseFloat energy) {
        cur_frame_++;
        bool succ = energy > energy_threshold_;
        switch (cur_status_) {
            case kSilence:
                if (succ) {
                    cur_step_++;
                    if (cur_step_ == disturb_frame_) {
                        start_frame_ = cur_frame_ - disturb_frame_;
                        cur_status_ = kActive;
                    }
                } else {
                    cur_step_ = 0;
                }
                break;
            case kActive:
                if (succ) {
                    cur_step_ = disturb_frame_;
                } else {
                    cur_step_--;
                    if (cur_step_ == 0) {
                        cur_status_ = kSilence;
                        segments_.push_back(std::make_pair(start_frame_, cur_frame_ - disturb_frame_));
                        KALDI_LOG << "[" << start_frame_ << ", " << cur_frame_ - disturb_frame_ << "]";
                    }
                }
                break;
            default:
                KALDI_ERR << "Unknown status existed";
        }
    }

    Segment &Result() {
        KALDI_LOG << "Process " << cur_frame_ << " frames";
        if (cur_status_ == kActive)
            segments_.push_back(std::make_pair(start_frame_, cur_frame_));
        return segments_;
    }


private:
    BaseFloat energy_threshold_;
    BaseFloat disturb_frame_;
    BaseFloat cur_step_;
    BaseFloat start_frame_, cur_frame_;
    Status cur_status_;
    Segment segments_;
};

int main(int argc, char *argv[]) {
    try {

        const char *usage =
                "Extract None silence segment from input wave, based on energy threshold\n" \
                "\n" \
                "Usage: wave-segment [options] <wav-rxfilename>\n" \
                "e.g. wave-segment template.wav\n";

        ParseOptions po(usage);

        BaseFloat energy_threshold = 5;
        BaseFloat frame_length_ms = 25.0;
        BaseFloat frame_shift_ms  = 10.0;
        BaseFloat disturb_frame = 10;

        po.Register("energy-threshold", &energy_threshold, "Per frame energy blow this value is"
                "regarded as silence frame, default 5");
        po.Register("frame-length", &frame_length_ms, "Frame length in milliseconds");
        po.Register("frame-shift" , &frame_shift_ms , "Frame shift in milliseconds");
        po.Register("disturb-frame" , &disturb_frame , "Number of the continuous disturbed frame");

        po.Read(argc, argv);

        if (po.NumArgs() != 1) {
            po.PrintUsage();
            exit(1);
        }

        std::string wave_in = po.GetArg(1);

        if (ClassifyRspecifier(wave_in, NULL, NULL) != kNoRspecifier)
            KALDI_ERR << "Input cannot be in rspecifier form";

        bool binary = true;
        Input wave(wave_in, &binary);
        WaveHolder holder;

        if (!holder.Read(wave.Stream()))
            KALDI_ERR << "Read failure from " << PrintableRxfilename(wave_in);

        WaveData wave_data_wrapper = holder.Value();
        Matrix<BaseFloat> wave_data = wave_data_wrapper.Data();

        BaseFloat sample_freq  = wave_data_wrapper.SampFreq(),
                frame_length = frame_length_ms / 1000 * sample_freq,
                frame_shift  = frame_shift_ms  / 1000 * sample_freq;

        SubVector<BaseFloat> waveform(wave_data, 0);
        VAD vad(energy_threshold, disturb_frame);

        int32 frame_start = 0, frame_end = waveform.Dim(), frame_id = 0;
        while (frame_start + frame_length < frame_end) {
            SubVector<BaseFloat> window = waveform.Range(frame_start, frame_length);
            BaseFloat energy = window.Norm(2) / frame_length;
            vad.Run(energy);
            frame_start += frame_shift;
            frame_id++;
        }

        std::vector<std::pair<int32, int32> > fix_segment = vad.Result();

        int32 start_samp, end_samp;
        char wave_name[32];

        for (int32 i = 0; i < fix_segment.size(); i++) {
            sprintf(wave_name, "%d-%d.wav", fix_segment[i].first, fix_segment[i].second);
            Output ko(std::string(wave_name), binary, false);
            start_samp = fix_segment[i].first * frame_shift;
            end_samp   = (fix_segment[i].second - fix_segment[i].first) * frame_shift + frame_length;
            SubMatrix<BaseFloat> segment_matrix = wave_data.Range(0, wave_data.NumRows(), start_samp, end_samp);
            WaveData writer(sample_freq, segment_matrix);
            writer.Write(ko.Stream());
            KALDI_LOG << "Write wave segments " << "[" << fix_segment[i].first << ", " << fix_segment[i].second << "] success";
        }

    } catch (const std::exception e) {
        std::cerr << e.what();
        return -1;
    }
}
