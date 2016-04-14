#include <iostream>
#include <vector>
#include <tuple>
#include <string>
#include <utility>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cmath>
#include <set>
#include "csv.h"

using namespace std;

class Dataset;
class DatsetView;

class Dataset {
public:
    Dataset(const vector<vector<int>>& data, 
        const vector<string>& headers): data(data), headers(headers) {
        vector<set<int>> values(cols());
        for (int i = 0; i < rows(); i++) {
            for (int j = 0; j < cols(); j++) {
                values[j].insert(data[i][j]);
            }
        }
        _n_values = vector<int>(cols(), 0);
        for (int i = 0; i < n_values.size(); i++) 
            _n_values[i] = values[i].size();

    }

    int rows() const {
        return data.size();
    }
    
    int cols() const {
        if (data.size() > 0) 
            return data[0].size();
        return 0;
    }

    int operator()(int i, int j) const {
        return data[i][j];
    }

    vector<int> operator()(int i) const {
        return data[i];
    }

    int n_values(int i) const {
        return _n_values[i];
    }

    const vector<vector<int>> data;
    const vector<string> headers;
    vector<int> _n_values;
};

class DatasetView {
public:
    DatasetView(const Dataset& dataset, const vector<bool>& selected): 
        dataset(dataset), selected(selected) { 
        for (int i = 0; i < dataset.size(); i++) {
            if (selected[i]) 
                idx.push_back(i);
        }   
    }

    int rows() const { return dataset.rows(); }
    int cols() const { return idx.size(); }

    int operator()(int i, int j) const {
        return dataset(i, idx[j]);
    }

    vector<int> operator()(int i) const {
        vector<int> result(cols(), 0);
        for (int j = 0; j < cols(); j++) 
            result[j] = dataset(i, idx[j]);
        return result;
    }

    int n_values(int i) const {
        return dataset.n_values(idx[i]);
    }

    const Dataset& dataset;
    vector<bool> selected;
    vector<int> idx;
};

class Factor {
    Factor(const DatasetView& dataset) { 
        int total_size = 1;
        for (int i = 0; i < dataset.cols(); i++) {
            stride.push_back(total_size);
            total_size *= dataset.n_values(i);
        }

        pvec.assign(total_size, 0);
        for (int i = 0; i < dataset.size(); i++) 
            p(dataset(i))++;
        normalize();
    }

    void normalize() {
        double Z = 0.;
        for (auto& x: pvec) 
            Z += x;
        for (auto& x: pvec) 
            x /= Z;
    }

    int n() const {
        return stride.size();
    }

    int _idx(const vector<int>& x) const {
        assert(x.size() == n());
        int idx = 0;
        for (int i = 0; i < x.size(); i++) {
            idx += x[i]*stride[i];
        }
        return idx;
    }

    int& p(const vector<int>& x) {
        return pvec[_idx(x)];
    }

    const int& p(const vector<int>& x) const {
        return pvec[_idx(x)];
    }

    double entropy() const {
        double result = 0.;
        for (int i = 0; i < pvec.size(); i++) 
            result += 
    }

    vector<int> stride; 
    vector<double> pvec;
};

class ScoreMI {
public:
    ScoreMI(const Dataset& dataset): dataset(dataset) { }

    double operator()(int x, vector<bool> pa) {
        vector<bool> temp(pa.size(), false);
        temp[x] = true;
        vector<double> p_xy factor_xy(dataset, pa);
        pa[x] = false;
        Factor factor_x(dataset, pa);
        double result = 0;
    }

    const Dataset& dataset;
};

class ScoreBIC {
public:
    ScoreBIC(const Dataset& dataset): dataset(dataset), mi(dataset) { }

    double operator()(int x, const vector<bool>& pa) {
                      
    }

    const Dataset& dataset;
    ScoreMI mi;
};

int main() {
    vector<string> headers;
    vector<vector<int>> data;
    tie(headers, data) = read_csv("data.csv");
    for (int i = 0; i < data.size(); i++) 
        data[i].erase(data[i].begin());
    headers.erase(headers.begin());

    Dataset dataset(data, headers);
    for (int i = 0; i < d.cols(); i++) {
        cout << d.headers[i] << ": "<< d.n_values[i] << endl;
    }
    ScoreMI score(dataset);
    return 0;
}
