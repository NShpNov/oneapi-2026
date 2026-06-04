#include "permutations_cxx.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

void Permutations(dictionary_t& dictionary) {
    using entry_t = dictionary_t::value_type;

    std::unordered_map<std::string, std::vector<entry_t*>> groups;
    groups.reserve(dictionary.size());

    for (auto& entry : dictionary) {
        entry.second.clear();

        std::string sorted = entry.first;
        std::sort(sorted.begin(), sorted.end());

        groups[std::move(sorted)].push_back(&entry);
    }

    for (auto& [_, members] : groups) {
        if (members.size() < 2) {
            continue;
        }

        for (entry_t* current : members) {
            auto& out = current->second;
            out.reserve(members.size() - 1);

            for (entry_t* other : members) {
                if (current != other) {
                    out.push_back(other->first);
                }
            }
        }
    }
}