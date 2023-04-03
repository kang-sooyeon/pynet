cmake_minimum_required(VERSION 3.4.1)
# 프로젝트 이름 및 버전
PROJECT ( "pynet" )
SET ( PROJECT_VERSION_MAJOR 0 )
SET ( PROJECT_VERSION_MINOR 1 )
 
# 빌드 형상(Configuration) 및 주절주절 Makefile 생성 여부
SET ( CMAKE_BUILD_TYPE Debug )
SET ( CMAKE_VERBOSE_MAKEFILE true )

include_directories(../include ../src ../../lib/arm64/libOpenCL/include ../../lib/arm64/libpng/include ../../lib/arm64/libzip/include)

SET ( SRC_FILES
        ../../src/device.cpp
        ../../src/memory.cpp
        ../../src/operation.cpp
        ../../src/main.cpp
        ../../src/maxPool2d.cpp
        ../../src/reflectionPad2d.cpp
        ../../src/cat.cpp
		../../src/sigmoid.cpp
		../../src/leakyReLU.cpp
		../../src/random.cpp
		../../src/dataLoader.cpp
		../../src/pngLoader.cpp
		../../src/convolution2d.cpp
		../../src/upsample.cpp
		../../src/instanceNorm2d.cpp
        ../../src/mseLoss.cpp
		../../src/upsampleConvLayer.cpp
		../../src/convLayer.cpp
		../../src/convMultiBlock.cpp
        ../../src/pynet.cpp
        ../../src/util.cpp
		../../src/svpng.inc
        )

SET ( OUTPUT_ELF
        "${CMAKE_PROJECT_NAME}-${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.out"
        )

add_library(libpng-lib STATIC IMPORTED)
add_library(libstdc++-lib SHARED IMPORTED)
add_library(libz-lib STATIC IMPORTED)
add_library(libOpenCL-lib SHARED IMPORTED)

set_target_properties(libstdc++-lib PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/arm64/libstdc++.so)
set_target_properties(libpng-lib PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/arm64/libpng/lib/libpng.a)
set_target_properties(libz-lib PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/arm64/libzip/lib/libz.a)
set_target_properties(libOpenCL-lib PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../../lib/arm64/libOpenCL/lib/libOpenCL.so)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

# 빌드 대상 바이너리 추가
ADD_EXECUTABLE( ${OUTPUT_ELF} ${SRC_FILES} )
target_link_libraries(${OUTPUT_ELF} libstdc++-lib libpng-lib libz-lib libOpenCL-lib)
