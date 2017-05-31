set(TARGET pyfbank)

set(SOURCE_FILES main.cpp segment-fbank-computer.cpp segment-fbank-computer.h)

add_library(${TARGET} SHARED ${SOURCE_FILES})
set_target_properties(${TARGET} PROPERTIES PREFIX "")
target_link_libraries(${TARGET} boost_python boost_numpy python2.7 pthread kaldi-util kaldi-base kaldi-matrix
        kaldi-transform kaldi-thread kaldi-feat)
