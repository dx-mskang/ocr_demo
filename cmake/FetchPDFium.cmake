# ========================================
# PDFium Build Configuration
# ========================================
# This file downloads and configures PDFium prebuilt binaries from:
# https://github.com/bblanchon/pdfium-binaries
#
# The pdfium-binaries repository should be added as a git submodule:
#   git submodule add https://github.com/bblanchon/pdfium-binaries.git 3rd-party/pdfium
#
# Since the repository only contains build scripts (not actual binaries),
# this CMake file downloads prebuilt binaries from GitHub Releases.
#
# Usage:
#   include(${CMAKE_SOURCE_DIR}/cmake/FetchPDFium.cmake)
#   target_link_libraries(my_target pdfium)

include(FetchContent)

# ========================================
# Configuration Options
# ========================================
# PDFium version - use chromium branch number
# Check available versions at: https://github.com/bblanchon/pdfium-binaries/releases
# TODO: 这里指定了 7630 的 pdfium 版本 
set(PDFIUM_VERSION "chromium/7630" CACHE STRING "PDFium version to download")

# Whether to enable V8 JavaScript engine (larger binary, more features)
option(PDFIUM_ENABLE_V8 "Enable PDFium with V8 JavaScript engine" OFF)

# PDFium root directory (submodule location)
set(PDFIUM_SUBMODULE_DIR "${CMAKE_SOURCE_DIR}/3rd-party/pdfium")

# Prebuilt binaries will be downloaded here
set(PDFIUM_PREBUILT_DIR "${PDFIUM_SUBMODULE_DIR}/prebuilt")

# ========================================
# Check Submodule (Optional but recommended)
# ========================================
if(EXISTS "${PDFIUM_SUBMODULE_DIR}/.git" OR EXISTS "${PDFIUM_SUBMODULE_DIR}/README.md")
    message(STATUS "PDFium submodule found at: ${PDFIUM_SUBMODULE_DIR}")
else()
    message(WARNING 
        "PDFium submodule not found at ${PDFIUM_SUBMODULE_DIR}\n"
        "Consider adding it with:\n"
        "  git submodule add https://github.com/bblanchon/pdfium-binaries.git 3rd-party/pdfium\n"
        "Continuing with prebuilt download only..."
    )
    # Create directory if it doesn't exist
    file(MAKE_DIRECTORY "${PDFIUM_SUBMODULE_DIR}")
endif()

# ========================================
# Platform Detection
# ========================================
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
        set(PDFIUM_PLATFORM "linux-x64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        set(PDFIUM_PLATFORM "linux-arm64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
        set(PDFIUM_PLATFORM "linux-arm")
    else()
        message(FATAL_ERROR "Unsupported Linux architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(PDFIUM_PLATFORM "win-x64")
    else()
        set(PDFIUM_PLATFORM "win-x86")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|ARM64")
        set(PDFIUM_PLATFORM "mac-arm64")
    else()
        set(PDFIUM_PLATFORM "mac-x64")
    endif()
else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

# ========================================
# Construct Download URL
# ========================================
# URL encode the version (replace / with %2F)
string(REPLACE "/" "%2F" PDFIUM_VERSION_ENCODED "${PDFIUM_VERSION}")

# Select V8 or non-V8 variant
if(PDFIUM_ENABLE_V8)
    set(PDFIUM_ARCHIVE_NAME "pdfium-v8-${PDFIUM_PLATFORM}.tgz")
else()
    set(PDFIUM_ARCHIVE_NAME "pdfium-${PDFIUM_PLATFORM}.tgz")
endif()

set(PDFIUM_DOWNLOAD_URL
    "https://github.com/bblanchon/pdfium-binaries/releases/download/${PDFIUM_VERSION_ENCODED}/${PDFIUM_ARCHIVE_NAME}"
)

message(STATUS "========================================")
message(STATUS "PDFium Configuration")
message(STATUS "========================================")
message(STATUS "  Version: ${PDFIUM_VERSION}")
message(STATUS "  Platform: ${PDFIUM_PLATFORM}")
message(STATUS "  V8 enabled: ${PDFIUM_ENABLE_V8}")
message(STATUS "  Submodule dir: ${PDFIUM_SUBMODULE_DIR}")
message(STATUS "  Prebuilt dir: ${PDFIUM_PREBUILT_DIR}")
message(STATUS "  Download URL: ${PDFIUM_DOWNLOAD_URL}")

# ========================================
# Check if prebuilt binaries already exist
# ========================================
set(PDFIUM_HEADER_FILE "${PDFIUM_PREBUILT_DIR}/include/fpdfview.h")
set(PDFIUM_NEED_DOWNLOAD TRUE)

if(EXISTS "${PDFIUM_HEADER_FILE}")
    # Check version file to see if we need to re-download
    set(PDFIUM_VERSION_FILE "${PDFIUM_PREBUILT_DIR}/VERSION")
    if(EXISTS "${PDFIUM_VERSION_FILE}")
        file(READ "${PDFIUM_VERSION_FILE}" PDFIUM_EXISTING_VERSION)
        string(STRIP "${PDFIUM_EXISTING_VERSION}" PDFIUM_EXISTING_VERSION)
        message(STATUS "  Existing version: ${PDFIUM_EXISTING_VERSION}")
        
        # Extract version number from PDFIUM_VERSION for comparison
        # e.g., "chromium/7630" -> compare with VERSION file content
        set(PDFIUM_NEED_DOWNLOAD FALSE)
        message(STATUS "  PDFium prebuilt binaries found, skipping download")
    endif()
endif()

# ========================================
# Download and Extract PDFium
# ========================================
if(PDFIUM_NEED_DOWNLOAD)
    message(STATUS "Downloading PDFium prebuilt binaries...")
    
    # Create prebuilt directory
    file(MAKE_DIRECTORY "${PDFIUM_PREBUILT_DIR}")
    
    # Download archive
    set(PDFIUM_ARCHIVE_PATH "${CMAKE_BINARY_DIR}/pdfium-download/${PDFIUM_ARCHIVE_NAME}")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/pdfium-download")
    
    if(NOT EXISTS "${PDFIUM_ARCHIVE_PATH}")
        message(STATUS "Downloading from: ${PDFIUM_DOWNLOAD_URL}")
        file(DOWNLOAD
            "${PDFIUM_DOWNLOAD_URL}"
            "${PDFIUM_ARCHIVE_PATH}"
            SHOW_PROGRESS
            STATUS DOWNLOAD_STATUS
            TLS_VERIFY ON
        )
        
        list(GET DOWNLOAD_STATUS 0 DOWNLOAD_ERROR_CODE)
        list(GET DOWNLOAD_STATUS 1 DOWNLOAD_ERROR_MESSAGE)
        
        if(NOT DOWNLOAD_ERROR_CODE EQUAL 0)
            file(REMOVE "${PDFIUM_ARCHIVE_PATH}")
            message(FATAL_ERROR 
                "Failed to download PDFium: ${DOWNLOAD_ERROR_MESSAGE}\n"
                "URL: ${PDFIUM_DOWNLOAD_URL}\n"
                "Please check your network connection and the version number."
            )
        endif()
    endif()
    
    # Extract archive to prebuilt directory
    message(STATUS "Extracting PDFium to: ${PDFIUM_PREBUILT_DIR}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E tar xzf "${PDFIUM_ARCHIVE_PATH}"
        WORKING_DIRECTORY "${PDFIUM_PREBUILT_DIR}"
        RESULT_VARIABLE EXTRACT_RESULT
    )
    
    if(NOT EXTRACT_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to extract PDFium archive")
    endif()
    
    message(STATUS "PDFium prebuilt binaries extracted successfully")
endif()

# ========================================
# Set PDFium paths
# ========================================
set(PDFIUM_ROOT "${PDFIUM_PREBUILT_DIR}")

# Find include directory
find_path(PDFium_INCLUDE_DIR
    NAMES "fpdfview.h"
    PATHS "${PDFIUM_ROOT}"
    PATH_SUFFIXES "include"
    NO_DEFAULT_PATH
)

# Read version from VERSION file
if(EXISTS "${PDFIUM_ROOT}/VERSION")
    file(READ "${PDFIUM_ROOT}/VERSION" PDFium_VERSION)
    string(STRIP "${PDFium_VERSION}" PDFium_VERSION)
else()
    set(PDFium_VERSION "${PDFIUM_VERSION}")
endif()

# ========================================
# Find and configure library
# ========================================
if(WIN32)
    # Windows: shared library (.dll + .lib)
    find_file(PDFium_LIBRARY_DLL
        NAMES "pdfium.dll"
        PATHS "${PDFIUM_ROOT}"
        PATH_SUFFIXES "bin"
        NO_DEFAULT_PATH
    )
    find_file(PDFium_LIBRARY_IMPLIB
        NAMES "pdfium.dll.lib"
        PATHS "${PDFIUM_ROOT}"
        PATH_SUFFIXES "lib"
        NO_DEFAULT_PATH
    )
    
    if(PDFium_LIBRARY_DLL AND PDFium_LIBRARY_IMPLIB)
        add_library(pdfium SHARED IMPORTED GLOBAL)
        set_target_properties(pdfium PROPERTIES
            IMPORTED_LOCATION "${PDFium_LIBRARY_DLL}"
            IMPORTED_IMPLIB "${PDFium_LIBRARY_IMPLIB}"
            INTERFACE_INCLUDE_DIRECTORIES "${PDFium_INCLUDE_DIR};${PDFium_INCLUDE_DIR}/cpp"
        )
        set(PDFium_LIBRARY "${PDFium_LIBRARY_DLL}")
        set(PDFium_FOUND TRUE)
    endif()
else()
    # Linux/macOS: shared library (.so / .dylib)
    find_library(PDFium_LIBRARY
        NAMES "pdfium"
        PATHS "${PDFIUM_ROOT}"
        PATH_SUFFIXES "lib"
        NO_DEFAULT_PATH
    )
    
    if(PDFium_LIBRARY)
        add_library(pdfium SHARED IMPORTED GLOBAL)
        set_target_properties(pdfium PROPERTIES
            IMPORTED_LOCATION "${PDFium_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${PDFium_INCLUDE_DIR};${PDFium_INCLUDE_DIR}/cpp"
        )
        set(PDFium_FOUND TRUE)
    endif()
endif()

# ========================================
# Verification and Status
# ========================================
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PDFium
    REQUIRED_VARS PDFium_LIBRARY PDFium_INCLUDE_DIR
    VERSION_VAR PDFium_VERSION
)

if(PDFium_FOUND)
    message(STATUS "========================================")
    message(STATUS "✓ PDFium configured successfully")
    message(STATUS "  Library: ${PDFium_LIBRARY}")
    message(STATUS "  Include: ${PDFium_INCLUDE_DIR}")
    message(STATUS "  Version: ${PDFium_VERSION}")
    message(STATUS "========================================")
    
    # Export variables for compatibility
    set(PDFIUM_INCLUDE_DIRS "${PDFium_INCLUDE_DIR}" CACHE INTERNAL "PDFium include directories")
    set(PDFIUM_LIBRARIES "${PDFium_LIBRARY}" CACHE INTERNAL "PDFium libraries")
    
    # For runtime library copying
    set(PDFIUM_RUNTIME_LIBRARY "${PDFium_LIBRARY}" CACHE INTERNAL "PDFium runtime library path")
else()
    message(FATAL_ERROR 
        "Failed to configure PDFium.\n"
        "Expected files at: ${PDFIUM_ROOT}\n"
        "Please check the download URL and network connection."
    )
endif()
