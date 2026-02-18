cmake_minimum_required(VERSION 3.16)
project(
  bch
  VERSION 0.1.0
  LANGUAGES CXX)

option(BCH_BUILD_TESTS "Build tests" ON)

add_library(bch lib/bch.cpp lib/gf.cpp)
add_library(bch::bch ALIAS bch)

target_compile_features(bch PUBLIC cxx_std_17)

target_include_directories(bch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(BCH_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

include(GNUInstallDirs)

install(
  TARGETS bch
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
