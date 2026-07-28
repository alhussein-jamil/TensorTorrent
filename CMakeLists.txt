cmake_minimum_required(VERSION 3.20)
project(streamcompiler_native LANGUAGES CXX)

option(STREAMCOMPILER_BUILD_NATIVE "Build native runtime helpers" OFF)

if(STREAMCOMPILER_BUILD_NATIVE)
  add_library(streamcompiler_runtime
    native/src/runtime_stub.cpp
  )
  target_include_directories(streamcompiler_runtime
    PUBLIC native/include
  )
  target_compile_features(streamcompiler_runtime PUBLIC cxx_std_17)
endif()
