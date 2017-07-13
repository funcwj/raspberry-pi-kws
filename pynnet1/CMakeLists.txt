set(CMAKE_CXX_STANDARD 11)
set(TARGET pynnet1)

set(SOURCE_FILES nnet-wrapper.cpp main.cpp nnet-wrapper.h)

add_library(${TARGET} SHARED ${SOURCE_FILES})
set_target_properties(${TARGET} PROPERTIES PREFIX "")
target_link_libraries(${TARGET} python2.7 boost_numpy boost_python pthread kaldi-base kaldi-util kaldi-matrix kaldi-cudamatrix kaldi-thread kaldi-nnet)
