#ifndef  TOKENIZER_H
#define  TOKENIZER_H


#include <regex>
#include <string>
#include <fstream>
#include <iostream>
#include "errors.h"
#include <vector>
#include <set>
#include <climits>
#include <unordered_map>
#include <codecvt>
#include "nlohmann/json.hpp"
#include <locale>

namespace tokenizer{

        class TiktokenFactory{
        public:
                std::ifstream file_;
                explicit TiktokenFactory(const std::string config);     
                std::string config_path;   
                std::string decode(std::vector<int> id);
                void encode(const std::string& str, std::vector<int>& ids);
        private:
                
                
        protected:
              
                struct hash_pair_wstring {
                size_t operator()(const std::pair<std::wstring, std::wstring>& p) const {
                auto hash1 = std::hash<std::wstring>{}(p.first);
                auto hash2 = std::hash<std::wstring>{}(p.second);
                // If hash1 == hash2, their XOR is zero.
                return (hash1 != hash2) ? hash1 ^ hash2 : hash1;
                 }
                };
                bool get_vocab();
                using BPERanks = std::unordered_map<std::pair<std::wstring, std::wstring>, int, hash_pair_wstring>;
                std::unordered_map<std::string, int> encoder_;
                BPERanks bpe_ranks_;
                std::vector<std::string> decoder_;
                std::unordered_map<uint8_t, wchar_t> b2u_;
                std::unordered_map<uint8_t, wchar_t> u2b_;
                void bpe(const std::wstring& token, const BPERanks& bpe_ranks, std::vector<std::wstring>* result);
                std::string decode_char(int id);
        };




}

#endif // ! TOKENIZER_H