//
// Created by wujian on 17-7-25.
//


#include "tpl-average.h"


typedef std::vector<std::pair<int32, int32> > alignment;

// default by size
int32 GetTemplateIndex(std::vector<Matrix<BaseFloat> > &posts) {
    KALDI_ASSERT(posts.size() >= 1);

    int32 index = 0;
    int32 min_length = posts[index].NumRows(), feats_length = posts[index].NumCols();

    for (int32 i = 1; i < posts.size(); i++) {
        KALDI_ASSERT(feats_length == posts[i].NumCols());
        if (posts[i].NumRows() < min_length) {
            index = i;
            min_length = posts[i].NumRows();
        }
    }
    return index;
}


BaseFloat LogCos(VectorBase<BaseFloat> &vec1, VectorBase<BaseFloat> &vec2) {
    BaseFloat dot_product = VecVec(vec1, vec2);
    BaseFloat mod_product = vec1.Norm(2) * vec2.Norm(2);
    return Log(dot_product / mod_product);
}

int32 MinimalIndex(BaseFloat *vec, int32 count) {
    BaseFloat min_element = vec[0];
    int32 index = 0;

    for (int i = 1; i < count; i++) {
        if (min_element > vec[i]) {
            min_element = vec[i];
            index = i;
        }
    }
    return index;
}

void GetAlignments(Matrix<BaseFloat> &t,
                   Matrix<BaseFloat> &q,
                   alignment *align) {
    Matrix<BaseFloat> dist(t.NumRows(), q.NumRows());
    Matrix<BaseFloat> step(t.NumRows(), q.NumRows());

    SubVector<BaseFloat> to = t.Row(0);
    SubVector<BaseFloat> qo = q.Row(0);
    dist(0, 0) = -LogCos(to, qo);
    // show start point
    step(0, 0) = 3;

    // ti
    for (int32 i = 1; i < dist.NumRows(); i++) {
        SubVector<BaseFloat> ti = t.Row(i);
        dist(i, 0) = dist(i - 1, 0) -LogCos(ti, qo);
        step(i, 0) = 2; // |
    }
    // qj
    for (int32 j = 1; j < dist.NumCols(); j++) {
        SubVector<BaseFloat> qj = q.Row(j);
        dist(0, j) = dist(j - 1, 0) -LogCos(to, qj);
        step(0, j) = 0; // --
    }

    for (int i = 1; i < dist.NumRows(); i++) {
        for (int j = 1; j < dist.NumCols(); j++) {
            SubVector<BaseFloat> ti = t.Row(i);
            SubVector<BaseFloat> qj = q.Row(j);
            BaseFloat cost[3] = {dist(i, j - 1), dist(i - 1, j - 1), dist(i - 1, j)};
            int32 src = MinimalIndex(cost, 3);
            KALDI_ASSERT(src >= 0 && src <= 2);
            step(i, j) = src;
            dist(i, j) = cost[src] - LogCos(ti, qj);
        }
    }

    int32 curx = dist.NumRows() - 1, cury = dist.NumCols() - 1;

    while(true) {
        align->push_back(std::make_pair(curx, cury));
        std::cout << "[" << curx << ", " << cury << "]" << std::endl;
        KALDI_ASSERT(curx >= 0 && cury >= 0);
        if (curx + cury == 0)
            break;
        int32 dir = static_cast<int32>(step(curx, cury));
        switch (dir) {
            case 0:
                cury--;
                break;
            case 1:
                curx--, cury--;
                break;
            case 2:
                curx--;
                break;
            default:
                KALDI_ERR << "Value dir must in [0, 2]";
                break;
        }
    }
    // reverse
    std::reverse(align->begin(), align->end());
    std::cout << "===============\n";
}


void TemplateAverage(std::vector<Matrix<BaseFloat> > &posts,
                     Matrix<BaseFloat> *avg) {
    int32 idx = GetTemplateIndex(posts);
    Matrix<BaseFloat> &t = posts[idx];
    avg->Resize(t.NumRows(), t.NumCols());

    Vector<BaseFloat> count(t.NumRows());

    for (int i = 0; i < posts.size(); i++) {

        // skip template
        if (i == idx)
            continue;

        Matrix<BaseFloat> &q = posts[i];
        alignment aligns;
        KALDI_LOG << "Align " << i << " with " << idx;
        GetAlignments(t, q, &aligns);

        for (int j = 0; j < aligns.size(); j++) {
            int32 tx = aligns[j].first, qx = aligns[j].second;
            KALDI_ASSERT(tx < t.NumRows() && tx >= 0);
            KALDI_ASSERT(qx < q.NumRows() && qx >= 0);
            count(tx) += 1;
            SubVector<BaseFloat> t_row(*avg, tx);
            t_row.AddVec(1, posts[i].Row(qx));
        }
    }

    for (int i = 0; i < avg->NumRows(); i++) {
        std::cout << i << ": " << count(i) << std::endl;
        avg->Row(i).Scale(1 / count(i));
    }
}
