#include "tokenizer.h"


using json = nlohmann::json;



std::wstring utf8_to_wstring(const std::string& str) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> myconv;
    return myconv.from_bytes(str);
}


std::string wstring_to_utf8(const std::wstring& str) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> myconv;
    return myconv.to_bytes(str);
}


tokenizer::tokenFactory::tokenFactory(const std::string config){
   
    file_.open(config);
    if (!file_.is_open()) {
        throw Error("failed to open file: " + config);
    }
    if(!get_vocab()){
        throw Error("load vocab is fail " );
    };
 
}

bool::tokenizer::tokenFactory::get_vocab(){

    auto json_data = json::parse(file_);
    auto vocab_json = json_data.at("model").at("vocab");
    auto merge_json = json_data.at("model").at("merges");
    std::size_t size = vocab_json.size();
    decoder_.resize(size);
    for (auto it = vocab_json.begin(); it!= vocab_json.end(); ++it) {
   
        std::string key = it.key();
        int value = it.value();
        encoder_[key] = value;
        decoder_[value]=key;

    }
    int i=0;
    for (const auto& element : merge_json) {

            std::string str_value= element.get<std::string>();
            int d = str_value.find(" ");
            bpe_ranks_.insert({{utf8_to_wstring(str_value.substr(0, d)),
                             utf8_to_wstring(str_value.substr(d + 1))}, i});
            i=i+1;
    }

    auto _insert_range = [=](int start, int end) {
        for (int c = start; c <= end; c++) {
            b2u_.insert({uint8_t(c), wchar_t(c)});       
        }
    };

    b2u_.clear();
 
    _insert_range(L'!', L'~');
    _insert_range(L'¡', L'¬');
    _insert_range(L'®', L'ÿ');
     int n = 0;
    for (int b = 0; b < 256; b++) {
        if (b2u_.find(uint8_t(b)) == b2u_.end()) {
            b2u_.insert({uint8_t(b), wchar_t(256 + n)});
            n++;
        }
    }
    for (auto e : b2u_) {
        u2b_.insert({e.second, e.first});
    }
    return true;
}
void get_pairs(const std::wstring& word, std::vector<std::pair<std::wstring, std::wstring>>* pairs) {
    pairs->clear();

    if (word.size() < 2) return;

    wchar_t previous = word[0];
    for (int i = 1; i < word.size(); i++) {
        pairs->push_back({std::wstring(1, previous), std::wstring(1, word[i])});
        previous = word[i];
    }
}

void tokenizer::tokenFactory::bpe(const std::wstring& token, const BPERanks& bpe_ranks, std::vector<std::wstring>* result) {
    std::set<int> merged;  // records indices in pairs that were merged.
    auto _left = [](int i, std::set<int>& merged) {
        for (int j = i - 1; j >= -1; j--) {
        if (merged.find(j) == merged.end()) return j;
        }
        return -1;
    };
    auto _right = [](int i, int cap, std::set<int>& merged) {
        for (int j = i + 1; j < cap; j++) {
        if (merged.find(j) == merged.end()) return j;
        }
        return cap;
    };

    std::vector<std::pair<std::wstring, std::wstring>> pairs;
    get_pairs(token, &pairs);

    while (true) {
        int min_score = INT_MAX;
        int to_merge = -1;  // indices into pairs.

        for (int i = 0; i < pairs.size(); ++i) {
        if (merged.find(i) == merged.end()) {  // pair i is not merged.
            auto iter = bpe_ranks.find(pairs[i]);
            int score = iter != bpe_ranks.end() ? iter->second : INT_MAX;
            if (score < min_score) {
            min_score = score;
            to_merge = i;
            }
        }
        }

        if (to_merge == -1) break;

        merged.insert(to_merge);
        std::wstring merge_into = pairs[to_merge].first + pairs[to_merge].second;

        int l = _left(to_merge, merged);
        if (l >= 0) pairs[l].second = merge_into;
        int r = _right(to_merge, pairs.size(), merged);
        if (r < pairs.size()) pairs[r].first = merge_into;
    }  // end while (true)

    if (merged.size() == pairs.size()) {
        result->push_back(token);

    } else {
        for (int i = 0; i < pairs.size(); ++i) {
            if (merged.find(i) == merged.end()) {
                if (_left(i, merged) < 0) result->push_back(pairs[i].first);
                result->push_back(pairs[i].second);
            }
        }
    }
}

void::tokenizer::tokenFactory::encode(const std::string& str, std::vector<int>& ids) {
    std::regex re("('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\\s\\w]+|\\s+)");
    std::string input = str;
    std::vector<std::string> result;
    std::string token;
    std::smatch match;
    while (std::regex_search(input, match, re)) {
        token = match.str(0);
        input = match.suffix().str();
        std::wstring wtoken;
        for (char c : token) {
            wtoken.push_back(b2u_.at(uint8_t(c)));
        }
        std::vector<std::wstring> bpe_tokens;
        bpe(wtoken, bpe_ranks_, &bpe_tokens);
        for (auto ws : bpe_tokens) {
            result.push_back(wstring_to_utf8(ws));
        }

    }
        for (auto s : result) {
        ids.push_back(encoder_.at(s));
    }
}


std::string tokenizer::tokenFactory::decode_char(int id) {
    
    if (id >= decoder_.size()) {
        return "";
    }
    std::wstring w = utf8_to_wstring(decoder_[id]);
    std::string r;
    for (wchar_t c : w) {
        if (u2b_.find(c) != u2b_.end()) {
            r.push_back(char(u2b_.at(c)));
        }
    }
    return r;
}

std::string tokenizer::tokenFactory::decode(std::vector<int> id){
    std::string  decode_string ;
    for (int i=0; i<id.size();i++)
    {
        decode_string+=decode_char(id[i]); 
    }

    return decode_string;
}

