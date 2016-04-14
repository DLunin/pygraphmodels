#ifndef _CSV_H_
#define _CSV_H_

#include <iostream>
#include <vector>
#include <tuple>
#include <string>
#include <utility>
#include <fstream>
#include <sstream>

using namespace std;

template <typename Container, typename Function>
auto fmap(const Container& container, Function f) -> vector<decltype(f(container[0]))> {
    typedef decltype(f(container[0])) ResultElement;
    vector<ResultElement> result;
    for (auto& item: container) {
        result.push_back(f(item)); 
    }
    return result;
}

vector<string> split(const string& str, const string& delim) {
    vector<string> result;
    for (int i = 0, j = str.find(delim, i); i < str.size(); i = j + delim.size(), j = str.find(delim, i)) {
        if (j == -1) j = str.size();
        result.push_back(str.substr(i, j - i));
    }
    return result;
}

tuple<vector<string>, vector<vector<int>>> read_csv(const string& filename) {
    ifstream istr(filename);
    const int BUF_SIZE = 1000;
    char buf[BUF_SIZE];
    istr.getline(buf, BUF_SIZE, '\n');
    auto headers = split(buf, ",");
    headers[0] = "id";
    vector<vector<int>> result;
    while (istr.getline(buf, BUF_SIZE, '\n')) {
        auto line = split(buf, ",");
        if (line.size() != headers.size())
            continue;
        result.push_back(fmap(line, [](string s){ return stoi(s); }));
    }
    return make_tuple(headers, result);
}

#endif
