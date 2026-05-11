add_executable(example1 "example1.cpp")

add_executable(example2 "example2.cpp")

add_executable(example3 "example3.cpp")

target_link_libraries(example1 PUBLIC backend)
target_link_libraries(example2 PUBLIC backend)
target_link_libraries(example3 PUBLIC backend)

