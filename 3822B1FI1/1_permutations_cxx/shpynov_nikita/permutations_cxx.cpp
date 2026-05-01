#include "permutations_cxx.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

void Permutations(dictionary_t& dictionary) {
    std::vector<std::string> chars;
    chars.reserve(dictionary.size());
    for (const auto& chr : dictionary) {
        chars.push_back(chr.first);
    }
    std::unordered_map<std::string, std::vector<std::string>> groups;
    groups.reserve(chars.size());
    for (const auto& s : chars) {
        std::string sorted = s;
        std::sort(sorted.begin(), sorted.end());
        groups[sorted].push_back(s);
    }
    for (auto& chr : dictionary) {
        std::string sorted = chr.first;
        std::sort(sorted.begin(), sorted.end());
        const auto& group = groups[sorted];
        for (const auto& member : group) {
            if (member != chr.first) {
                chr.second.push_back(member);
            }
        }
    }
}
