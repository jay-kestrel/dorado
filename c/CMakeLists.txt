cmake_minimum_required(VERSION 3.16)
project(dorado_c C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

# The port links only libm: zero external dependencies (spec, locked
# decision 5).
find_library(M_LIBRARY m)
mark_as_advanced(M_LIBRARY)

set(CORE_SOURCES src/mt19937.c src/npy.c src/json.c src/field.c src/walker.c)
set(CORE_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(c_port_core STATIC ${CORE_SOURCES})
target_include_directories(c_port_core PUBLIC ${CORE_INCLUDES})
if(M_LIBRARY)
  target_link_libraries(c_port_core PUBLIC ${M_LIBRARY})
endif()

add_executable(dorado_c src/main.c)
target_link_libraries(dorado_c PRIVATE c_port_core)

enable_testing()

function(c_port_test name)
  add_executable(${name} tests/${name}.c)
  target_link_libraries(${name} PRIVATE c_port_core)
  target_compile_definitions(${name} PRIVATE
    DORADO_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/..")
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES TIMEOUT 120)
endfunction()

c_port_test(test_mt19937)
c_port_test(test_npy_json)
c_port_test(test_weight)
c_port_test(test_travel_times)
c_port_test(test_trajectory)

# end-to-end: run the driver binary on the 120x240 fixture, then diff the
# walk-data JSON against the committed golden (produced by
# benchmarks/run_reference.py; see c/tests/golden/README)
add_test(NAME e2e_driver
  COMMAND dorado_c
    --input ${CMAKE_CURRENT_SOURCE_DIR}/../bench/fixture_rcm
    --particles 200 --iterations 5 --seed 0
    --out ${CMAKE_CURRENT_BINARY_DIR}/e2e_out)
set_tests_properties(e2e_driver PROPERTIES TIMEOUT 120)

add_executable(test_end_to_end tests/test_end_to_end.c)
target_link_libraries(test_end_to_end PRIVATE c_port_core)
add_test(NAME e2e_compare COMMAND test_end_to_end
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/golden/walk_data_rcm_s200_i5.json
  ${CMAKE_CURRENT_BINARY_DIR}/e2e_out/walk_data.json)
set_tests_properties(e2e_compare PROPERTIES TIMEOUT 60
  DEPENDS e2e_driver)
