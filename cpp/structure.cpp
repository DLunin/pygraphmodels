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
#include "mdarray.h"

using namespace std;

class Dataset;
class DatsetView;

class Dataset {
public:
    Dataset(const vector<vector<int>>& data_, 
        const vector<string>& headers): data({(int)data_.size(), (int)data_[0].size()}), headers(headers) {
        for (int i = 0; i < data_.size(); i++) {
            for (int j = 0; j < data_[i].size(); j++) 
                const_cast<int&>(data[{i, j}]) = data_[i][j];
        }

        vector<set<int>> values(cols());
        for (int i = 0; i < rows(); i++) {
            for (int j = 0; j < cols(); j++) {
                values[j].insert(data_[i][j]);
            }
        }
        _n_values = vector<int>(cols(), 0);
        for (int i = 0; i < _n_values.size(); i++) 
            _n_values[i] = values[i].size();

    }

    int rows() const {
        return data.shape(0);
    }
    
    int cols() const {
        return data.shape(1);
    }

    int operator()(int i, int j) const {
        return data[{i, j}];
    }

    const array_1d<int> operator()(int i) const {
        return array_1d<int>(const_cast<int*>(data.ptr_to({i})), cols());
    }

    int n_values(int i) const {
        return _n_values[i];
    }

    const array_nd<int> data;
    const vector<string> headers;
    vector<int> _n_values;
};

//class DatasetView {
//public:
    //DatasetView(const Dataset& dataset, const vector<bool>& selected): 
        //dataset(dataset), selected(selected) { 
        //for (int i = 0; i < dataset.rows(); i++) {
            //if (selected[i])
                //idx.push_back(i);
        //}   
    //}

    //int rows() const { return dataset.rows(); }
    //int cols() const { return idx.size(); }

    //int operator()(int i, int j) const {
        //return dataset(i, idx[j]);
    //}

    //vector<int> operator()(int i) const {
        //vector<int> result(cols(), 0);
        //for (int j = 0; j < cols(); j++) 
            //result[j] = dataset(i, idx[j]);
        //return result;
    //}

    //int n_values(int i) const {
        //return dataset.n_values(idx[i]);
    //}

    //const Dataset& dataset;
    //vector<bool> selected;
    //vector<int> idx;
//};

class Factor {
public:
    Factor(const Dataset& dataset, array_1d<bool> vars): 
        prob([dataset, vars](){
            array_1d<int> shape(vars.size());
            for (int i = 0; i < shape.size(); i++) {
                if (vars(i)) 
                    shape(i) = dataset.n_values(i);
                else shape(i) = 1;
            }
            return shape;
        }())

        {
            for (int i = 0; i < dataset.rows(); i++) {
                prob[dataset(i)[vars]] += 1. / dataset.rows();     
            }
        }
    
    ~Factor() {   }
public:
    array_nd<double> prob;  
};

class ScoreMI {
public:
    ScoreMI(const Dataset& dataset): dataset(dataset) { }

    double operator()(int x, vector<bool> pa) {
        //vector<bool> temp(pa.size(), false);
        //temp[x] = true;

        //pa[x] = false;
        //Factor factor_x(dataset, pa);
        //double result = 0;
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
    for (int i = 0; i < dataset.cols(); i++) {
        cout << dataset.headers[i] << ": " << dataset.n_values(i) << endl;
    }

    array_1d<bool> vars(dataset.cols());
    vars(0) = true;

    Factor fact(dataset, vars);
    cout << *fact.prob.begin() << endl;
    return 0;
}
