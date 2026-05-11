#pragma once
#include <filesystem>
#include <string>

inline std::string getDataPath() {
	auto str = std::filesystem::current_path().string() + "/../../../../examples/data/";
	for (auto& c : str) {
		if (c == '\\') {
			c = '/';
		}
	}
	return str;
}