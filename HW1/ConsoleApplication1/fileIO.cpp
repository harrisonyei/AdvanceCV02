#include "fileIO.h"

std::vector<std::string> get_filenames_in(fs::path path, std::string target_subtitle)
{
    std::vector<std::string> filenames;

    // http://en.cppreference.com/w/cpp/experimental/fs/directory_iterator
    const fs::directory_iterator end{};

    std::cout << "Loading files..." << std::endl;

    for (fs::directory_iterator iter{ path }; iter != end; ++iter)
    {
        // http://en.cppreference.com/w/cpp/experimental/fs/is_regular_file
        if (fs::is_regular_file(*iter)) {
            std::string filename = iter->path().string();
            std::string subtitle = filename.substr(filename.size() - target_subtitle.size(), target_subtitle.size());
            if (subtitle == target_subtitle) {
                filenames.push_back(filename);

                std::cout << filename << std::endl;
            }
        }
    }

    std::sort(filenames.begin(), filenames.end());
    std::cout << std::endl;

    return filenames;
}
