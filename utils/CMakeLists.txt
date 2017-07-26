cmake_minimum_required(VERSION 3.5)
project(WaveSegment)

set(CMAKE_CXX_STANDARD 11)
set(KALDI_DIR /home/wujian/Document/git/kaldi)
add_definitions(-O3 -g -std=c++11 -DHAVE_CLAPACK)
include_directories(${KALDI_DIR}/tools/CLAPACK ${KALDI_DIR}/src)
link_directories(${KALDI_DIR}/src/lib)

# add_executable(wave-segment wave-segment.cpp tpl-average-test.cpp)
add_executable(tpl-average-test tpl-average.cpp tpl-average-test.cpp tpl-average.h)

# target_link_libraries(wave-segment pthread kaldi-base kaldi-util kaldi-matrix kaldi-cudamatrix kaldi-thread kaldi-feat)
target_link_libraries(tpl-average-test pthread kaldi-base kaldi-util kaldi-matrix kaldi-cudamatrix kaldi-thread kaldi-feat)

