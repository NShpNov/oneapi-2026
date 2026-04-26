#include "permutations_cxx.h"

#include <algorithm>
#include <array>
#include <unordered_map>

void Permutations(dictionary_t &dictionary) {
  std::unordered_map<std::string, std::vector<dictionary_t::iterator>> anagramGroups;
  anagramGroups.reserve(dictionary.size());

  for (auto entryIt = dictionary.begin(); entryIt != dictionary.end(); ++entryIt) {
    std::string sortedKey = entryIt->first;
    std::sort(sortedKey.begin(), sortedKey.end());
    anagramGroups[sortedKey].push_back(entryIt);
  }

  for (auto &[sortedKey, entryGroup] : anagramGroups) {
    if (entryGroup.size() <= 1)
      continue;

    std::vector<std::string_view> sortedWords;
    sortedWords.reserve(entryGroup.size());
    for (auto groupIt : entryGroup) {
      sortedWords.push_back(groupIt->first);
    }
    std::sort(sortedWords.begin(), sortedWords.end(), std::greater<std::string_view>());

    for (auto &groupIt : entryGroup) {
      std::vector<std::string> otherWords;
      otherWords.reserve(sortedWords.size() - 1);
      for (const auto &word : sortedWords) {
        if (word != groupIt->first)
          otherWords.push_back(std::string(word));
      }
      groupIt->second = std::move(otherWords);
    }
  }
}