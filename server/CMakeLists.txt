# ========================================
# Server CMakeLists.txt
# ========================================
# HTTP API Server for DeepX OCR
cmake_minimum_required(VERSION 3.15)

# ========================================
# Dependencies
# ========================================
# CURL for HTTP client
find_package(CURL REQUIRED)

# UUID library (via pkg-config)
find_package(PkgConfig REQUIRED)
pkg_check_modules(UUID REQUIRED uuid)

# ========================================
# PDFium Configuration (PDF rendering library)
# ========================================
# PDFium is managed as a submodule at 3rd-party/pdfium
# Prebuilt binaries are downloaded from GitHub Releases
# See: https://github.com/bblanchon/pdfium-binaries
option(BUILD_WITH_PDFIUM "Build with PDFium support for PDF processing" ON)
if(BUILD_WITH_PDFIUM)
    include(${CMAKE_SOURCE_DIR}/cmake/FetchPDFium.cmake)
endif()

# ========================================
# Server Sources
# ========================================
set(SERVER_SOURCES
    server_main.cpp
    ocr_handler.cpp
    json_response.cpp
    file_handler.cpp
    pdf_handler.cpp
    ${CMAKE_SOURCE_DIR}/3rd-party/cpp-base64/base64.cpp
)

# Create server executable
add_executable(ocr_server ${SERVER_SOURCES})

# ========================================
# Include Directories
# ========================================
target_include_directories(ocr_server PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/3rd-party/crow/include
    ${CMAKE_SOURCE_DIR}/3rd-party/cpp-base64
    ${CMAKE_SOURCE_DIR}/3rd-party/json/include
    ${CMAKE_SOURCE_DIR}/3rd-party/spdlog/include
    ${OpenCV_INCLUDE_DIRS}
    ${UUID_INCLUDE_DIRS}
)

# ========================================
# Link Libraries
# ========================================
set(SERVER_LINK_LIBS
    ocr_pipeline
    ocr_detection
    ocr_classification
    ocr_recognition
    ocr_preprocessing
    ocr_common
    ${OpenCV_LIBS}
    CURL::libcurl
    ${UUID_LIBRARIES}
    pthread
)

# Add PDFium if enabled
if(BUILD_WITH_PDFIUM AND PDFium_FOUND)
    list(APPEND SERVER_LINK_LIBS pdfium)
endif()

target_link_libraries(ocr_server PRIVATE ${SERVER_LINK_LIBS})

# ========================================
# Target Properties
# ========================================
set_target_properties(ocr_server PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ========================================
# Post-build: Copy PDFium runtime library
# ========================================
if(BUILD_WITH_PDFIUM AND PDFium_FOUND)
    add_custom_command(TARGET ocr_server POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${PDFIUM_RUNTIME_LIBRARY}"
            "$<TARGET_FILE_DIR:ocr_server>"
        COMMENT "Copying PDFium library to output directory"
    )
endif()

# ========================================
# Install Rules
# ========================================
install(TARGETS ocr_server DESTINATION bin)

if(BUILD_WITH_PDFIUM AND PDFium_FOUND)
    install(FILES "${PDFIUM_RUNTIME_LIBRARY}" DESTINATION lib)
endif()

# ========================================
# Status Messages
# ========================================
message(STATUS "========================================")
message(STATUS "Server Configuration")
message(STATUS "========================================")
message(STATUS "  Executable: ocr_server")
message(STATUS "  CURL: ${CURL_LIBRARIES}")
message(STATUS "  UUID: ${UUID_LIBRARIES}")
if(BUILD_WITH_PDFIUM AND PDFium_FOUND)
    message(STATUS "  PDFium: ${PDFium_LIBRARY}")
else()
    message(STATUS "  PDFium: DISABLED")
endif()
message(STATUS "========================================")
