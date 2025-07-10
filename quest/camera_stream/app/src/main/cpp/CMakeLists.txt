cmake_minimum_required(VERSION 3.22)
project(streamer)

add_library(streamer SHARED native_passthrough_streamer.cpp)

# Quest 3 is 64-bit only
set(CMAKE_ANDROID_ARCH_ABI arm64-v8a)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# NDK libraries
find_library(log-lib        log)
find_library(android-lib    android)
find_library(media-lib      mediandk)
find_library(camera2ndk-lib camera2ndk)

target_link_libraries(streamer
        ${log-lib}
        ${android-lib}
        ${media-lib}
        ${camera2ndk-lib})
