find_package(Boost REQUIRED COMPONENTS unit_test_framework)

file(GLOB_RECURSE TEST_FILES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

foreach(test_file IN LISTS TEST_FILES)
  get_filename_component(test_name "${test_file}" NAME_WE)

  add_executable("${test_name}" "${test_file}")
  target_link_libraries("${test_name}" PRIVATE bch::bch
                                               Boost::unit_test_framework)

  target_compile_definitions("${test_name}" PRIVATE BOOST_TEST_DYN_LINK
                                                    BOOST_TEST_MAIN)

  add_test(NAME "${test_name}" COMMAND "${test_name}")
endforeach()
