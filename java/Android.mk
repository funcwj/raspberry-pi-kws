LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)
LOCAL_MODULE := kaldi-prebuild
LOCAL_SRC_FILES := libkaldi.a
# LOCAL_CFLAGS := -DSkip_f2c_Undefs
include $(PREBUILT_STATIC_LIBRARY)

include $(CLEAR_VARS)

LOCAL_MODULE    := kaldi
LOCAL_SRC_FILES := impl.cpp evaluator.cpp tpl-average.cpp

LOCAL_CFLAGS += -std=c++11 -DKALDI_DOUBLEPRECISION=0 -DHAVE_CXXABI_H \
                 -DHAVE_OPENBLAS -DANDROID_BUILD -ftree-vectorize \
                 -mfloat-abi=hard -mfpu=neon -mhard-float -D_NDK_MATH_NO_SOFTFP=1

LOCAL_LDFLAGS += -Wl,--no-warn-mismatch -lm_hard

LOCAL_STATIC_LIBRARIES += kaldi-prebuild

LOCAL_C_INCLUDES += $(LOCAL_PATH)/blas $(LOCAL_PATH)
LOCAL_LDLIBS += -llog

# LOCAL_CPP_FEATURES += exceptions
# LOCAL_CFLAGS += -fexceptions

LOCAL_ALLOW_UNDEFINED_SYMBOLS := true
include $(BUILD_SHARED_LIBRARY)
