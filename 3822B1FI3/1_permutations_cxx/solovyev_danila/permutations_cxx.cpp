#include "permutations_cxx.h"
#include <functional>
#include <algorithm>

void Permutations(dictionary_t& dictionary) {
    std::map<std::string, std::vector<std::string>> groups;
    for (const auto& pair : dictionary) {
        std::string key = pair.first;
        std::string sortedKey = key;
        std::sort(sortedKey.begin(), sortedKey.end());
        groups[sortedKey].push_back(key);
    }
    for (auto& pair : dictionary) {
        std::vector<std::string> result;
        std::string key = pair.first;
        std::string sortedKey = key;
        std::sort(sortedKey.begin(), sortedKey.end());
        const std::vector<std::string>& group = groups[sortedKey];
        for (const auto& possibleKey : group) {
            if (possibleKey != key) {
                result.push_back(possibleKey);
            }
        }
        std::sort(result.begin(), result.end(), std::greater<std::string>());
        pair.second = result;
    }
}