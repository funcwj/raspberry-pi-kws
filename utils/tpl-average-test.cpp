//
// Created by wujian on 17-7-26.
//

#include "util/common-utils.h"
#include "tpl-average.h"


int main() {
    std::vector<Matrix<BaseFloat> > post;
    SequentialBaseFloatMatrixReader kaldi_reader(std::string("ark:templs.post"));
    for (; !kaldi_reader.Done(); kaldi_reader.Next()) {
        post.push_back(kaldi_reader.Value());
    }
    Matrix<BaseFloat> avg;
    TemplateAverage(post, &avg);
    std::cout << "Done!" << std::endl;
    return 0;
}