#include "permutations_cxx.h"
#include <algorithm>
#include <map>

void Permutations(dictionary_t& dictionary) {
    std::map<std::string, std::vector<std::string>> groups;

    for (const auto& pair : dictionary) {
        std::string word = pair.first;
        std::string key = word;

        std::sort(key.begin(), key.end());
        groups[key].push_back(word);
    }

    for (auto& pair : dictionary) {
        const std::string& word = pair.first;

        std::string key = word;
        std::sort(key.begin(), key.end());

        std::vector<std::string> permutations;

        for (const auto& candidate : groups[key]) {
            if (candidate != word) {
                permutations.push_back(candidate);
            }
        }
        std::sort(permutations.begin(), permutations.end(), std::greater<std::string>());

        pair.second = std::move(permutations);
    }
}