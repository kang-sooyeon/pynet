#! /bin/bash
rm -r CMakeFiles
rm CMakeCache.txt
rm cmake_install.cmake
rm Makefile

export NDK=CURRENT_NDK_PATH
export ABI=arm64-v8a

export CXX=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++
export CC=$NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang

cmake \
    -DCMAKE_ANDROID_NDK=$NDK \
    -DCMAKE_SYSTEM_NAME=Android \
    -DCMAKE_ANDROID_ARCH_ABI=arm64-v8a \
    -DCMAKE_TOOLCHAIN_FILE=$NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=$ABI \
    -DANDROID_NATIVE_API_LEVEL=29 \
    -DCMAKE_CXX_FLAGS="-Wall -O3" \
    -DCMAKE_CXX_COMPILER=$CXX \
    ..
