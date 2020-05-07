#pragma once

#include <iostream>
#include <vector>
#include <string>
// #include <ifstream>

//#define DEBUG

#ifndef DEBUG // microsoft compiler/library, windows

#include <experimental/filesystem> // http://en.cppreference.com/w/cpp/experimental/fs
namespace fs = std::experimental::filesystem;
const std::string test_directory = "C:\\windows";

#else // #ifdef _GNUG_ // may be GNU compiler, unix or unix-clone

#include <filesystem>

namespace fs = std::filesystem;

#endif // microsoft/GNU

std::vector<std::string> get_filenames_in(fs::path path, std::string target_subtitle);

