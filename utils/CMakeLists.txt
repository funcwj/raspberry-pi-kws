set(CMAKE_CXX_STANDARD 11)
set(TARGET wave-segment)

set(SOURCE_FILES wave-segment.cpp)
add_executable(${TARGET} ${SOURCE_FILES})
target_link_libraries(${TARGET} kaldi-base kaldi-util kaldi-matrix kaldi-cudamatrix kaldi-feat)
