cmake_minimum_required(VERSION 3.16)
project(fitting_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(use_logilkelihood use_loglikelihood.cpp)
