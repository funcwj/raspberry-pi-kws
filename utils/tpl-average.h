//
// Created by wujian on 17-7-26.
//

#ifndef TPL_AVERAGE_H
#define TPL_AVERAGE_H

#include "cudamatrix/cu-matrix.h"

using namespace kaldi;

void TemplateAverage(std::vector<Matrix<BaseFloat> > &posts,
                     Matrix<BaseFloat> *avg);

#endif //TPL_AVERAGE_H
