# Custom cmake target for running nosetests on compiled libraries.
#
# This creates a custom target for running nosetests with the intent of
# verifying a compiled library using interpreted Python unit tests. It is
# meant to be used in addition to, not in lieu of, compiled unit test code.
# This can be particularly useful for math-heavy code where a compiled unit
# test could be quite cumbersome. The Python test code is responsible for
# importing the libraries (probably using ctypes).
#
# Usage:
#
#   add_python_test_target(TARGET_NAME LIBRARY_DEPENDENCY SOURCE_FILES)
#
# Released into the public domain. No warranty implied.

find_program(NOSETESTS_PATH nosetests3)
if(NOT NOSETESTS_PATH)
    message(WARNING
        "nosetests not found! Python library tests will not be available.")
endif()


function(add_python_test_target TARGET_NAME TARGET_LIB)
    # Try again to find nosetests here. We may have switched virtualenvs or
    # something since first running cmake.
    find_program(NOSETESTS_PATH nosetests3)
    if(NOT NOSETESTS_PATH)
        message(FATAL_ERROR "nosetests not found! Aborting...")
    endif()

    set(COPY_DIR ${CMAKE_BINARY_DIR}/${TARGET_NAME}_files)

    add_custom_target(${TARGET_NAME}
        COMMAND ${NOSETESTS_PATH}
        WORKING_DIRECTORY ${COPY_DIR}
        COMMENT "Running Python tests.")

    # Copy Python files to the local binary directory so they can find the
    # dynamic library.
    set(COPY_TARGET ${TARGET_NAME}_copy)

    # We add a separate target for copying Python files. This way we can set
    # it as a dependency for the nosetests target.
    add_custom_target(${COPY_TARGET})

    # Make sure the directory exists before we copy the files.
    add_custom_command(TARGET ${COPY_TARGET}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${COPY_DIR})

    # Add commands to copy each file in the list of sources.
    foreach(pysource ${ARGN})
        add_custom_command(TARGET ${COPY_TARGET}
            COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_CURRENT_SOURCE_DIR}/${pysource} ${COPY_DIR})
    endforeach()

    # Add a command to copy the target library into the same folder as the
    # python files.
    add_custom_command(TARGET ${COPY_TARGET}
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:${TARGET_LIB}> ${COPY_DIR})

    # Make the copy target a dependency of the testing target to ensure it
    # gets done first.
    add_dependencies(${TARGET_NAME} ${COPY_TARGET})
    add_dependencies(${COPY_TARGET} ${TARGET_LIB})
endfunction()
