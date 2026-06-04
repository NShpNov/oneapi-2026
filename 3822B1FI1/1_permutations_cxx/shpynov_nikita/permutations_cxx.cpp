#include "permutations_cxx.h"
#include <iterator>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

void Permutations(dictionary_t& dictionary) {
    using words_t = std::vector<std::string>;

    std::unordered_map<std::string, words_t> groups;
    groups.reserve(dictionary.size());

    for (const auto& [word, _] : dictionary) {
        std::string sorted = word;
        std::sort(sorted.begin(), sorted.end());

        groups[std::move(sorted)].push_back(word);
    }

    for (auto& [_, group] : groups) {
        if (group.size() <= 1) continue;

        std::sort(group.begin(), group.end(), std::greater<std::string>());
    }

    for (auto& [word, permutations] : dictionary) {
        std::string sorted = word;
        std::sort(sorted.begin(), sorted.end());

        auto it = groups.find(sorted);
        if (it == groups.end() || it->second.size() <= 1) continue;

        const auto& group = it->second;
        permutations.clear();
        permutations.reserve(group.size() - 1);
        
        std::copy_if(group.begin(), group.end(), 
                     std::back_inserter(permutations),
                     [&word](const std::string& cand) { return cand != word; });
    }
}