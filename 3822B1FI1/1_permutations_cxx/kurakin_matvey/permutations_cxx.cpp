#include "permutations_cxx.h"

#include <algorithm>
#include <array>
#include <unordered_map>

void Permutations(dictionary_t &dictionary) {
  std::unordered_map<std::string, std::vector<dictionary_t::iterator>> groups;
  groups.reserve(dictionary.size());

  for (auto it = dictionary.begin(); it != dictionary.end(); ++it) {
    std::string key = it->first;
    std::sort(key.begin(), key.end());
    groups[key].push_back(it);
  }

  for (auto &[key, group] : groups) {
    if (group.size() <= 1)
      continue;

    std::vector<std::string_view> perms;
    perms.reserve(group.size());
    for (auto it : group) {
      perms.push_back(it->first);
    }
    std::sort(perms.begin(), perms.end(), std::greater<std::string_view>());

    for (auto &it : group) {
      std::vector<std::string> list;
      list.reserve(perms.size() - 1);
      for (const auto &s : perms) {
        if (s != it->first)
          list.push_back(std::string(s));
      }
      it->second = std::move(list);
    }
  }
}