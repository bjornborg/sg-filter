
cmake_minimum_required(VERSION 3.10)

project(sg-filter)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT WIN32)
  set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libstdc++")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} \
      -D_FORTIFY_SOURCE=2 -O3 -s")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} \
      -g" )
# -fsanitize=thread
  # -fsanitize=address
  set(CMAKE_CXX_FLAGS
      "${CMAKE_CXX_FLAGS} \
      -D_XOPEN_SOURCE=700 -fstack-protector \
      -fomit-frame-pointer -pipe -pedantic -pedantic-errors -Werror -Weffc++ \
      -Wall -Wextra -Wshadow -Wdeprecated -Wdiv-by-zero -Wfloat-equal \
      -Wfloat-conversion -Wsign-compare -Wpointer-arith -Wuninitialized \
      -Wunreachable-code -Wunused -Wunused-function -Wunused-label \
      -Wunused-parameter -Wunused-but-set-parameter -Wunused-but-set-variable \
      -Wunused-value -Wunused-variable -Wunused-result \
      -Wmissing-field-initializers -Wmissing-format-attribute \
      -Wmissing-include-dirs -Wmissing-noreturn")
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty)

# ##############################################################################
# Eigen3 lib
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})
set(LIBRARIES ${LIBRARIES} ${EIGEN3_LIBRARIES})



# ##############################################################################
# gnuplot-iostream lib
# find_package(Boost REQUIRED COMPONENTS iostreams system filesystem)
# include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/gnuplot-iostream)
# set(GnuPlotIostream_BuildTests
#     OFF
#     CACHE INTERNAL "build gnuplot iostream tests" FORCE)
# set(GnuPlotIostream_BuildExamples
#     OFF
#     CACHE INTERNAL "build gnuplot iostream examples" FORCE)
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/gnuplot-iostream)
# set(LIBRARIES ${LIBRARIES} gnuplot_iostream Boost::iostreams Boost::system
#               Boost::filesystem)


add_library(
  filter-core OBJECT
  ${CMAKE_CURRENT_SOURCE_DIR}/src/sg-filter.cpp)

# $<TARGET_OBJECTS:tinyso-core>

set(LIBRARIES ${LIBRARIES} Threads::Threads)

# enable_testing()
find_package(Catch2 3 REQUIRED)

add_executable(
  ${PROJECT_NAME}-runner ${CMAKE_CURRENT_SOURCE_DIR}/test/test-sg-filter.cpp
  $<TARGET_OBJECTS:filter-core>
)
target_link_libraries(${PROJECT_NAME}-runner PRIVATE  ${LIBRARIES} Catch2::Catch2WithMain)
add_test(NAME ${PROJECT_NAME}-ga-runner COMMAND ${PROJECT_NAME}-ga-runner)
