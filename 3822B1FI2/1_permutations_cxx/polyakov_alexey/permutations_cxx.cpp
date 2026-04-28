#include "permutations_cxx.h"

#include <unordered_map>
#include <array>
#include <algorithm>
#include <string>

static std::string make_key(const std::string& word) {
    std::array<int, 26> freq = { 0 };

    for (char c : word) {
        freq[c - 'a']++;
    }

    std::string key;
    key.reserve(26 * 2);

    for (int f : freq) {
        key += std::to_string(f);
        key += '#';
    }

    return key;
}

void Permutations(dictionary_t& dictionary) {
    std::unordered_map<std::string, std::vector<std::string>> groups;
    std::unordered_map<std::string, std::string> cache;

    for (const auto& [word, _] : dictionary) {
        auto key = make_key(word);
        cache[word] = key;
        groups[key].push_back(word);
    }

    for (auto& [word, vec] : dictionary) {
        const auto& key = cache[word];
        const auto& group = groups[key];

        vec.clear();

        vec.reserve(group.size() > 0 ? group.size() - 1 : 0);

        for (const auto& other : group) {
            if (other != word) {
                vec.push_back(other);
            }
        }

        std::sort(vec.rbegin(), vec.rend());
    }
}