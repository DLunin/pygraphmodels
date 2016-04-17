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
        const vector<string>& headers): data({(int)data_.size(), (int)data_[0].size()}), headers(headers), n_values((int)data_[0].size()) {
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
        for (int i = 0; i < n_values.size(); i++) 
            n_values(i) = values[i].size();

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

    const array_nd<int> data;
    const vector<string> headers;
public:
    array_1d<int> n_values;
};

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
                //cout << dataset(i) << endl;
                //for (auto& p: prob) 
                    //cout << p << " ";
                //cout << endl;
                prob[dataset(i)] += 1. / dataset.rows();
            }
        }

    Factor(const array_nd<double>& p): prob(p) {
             
    }

    void normalize() {
        double sum = accumulate(prob.begin(), prob.end(), 0.);
        for (auto it = prob.begin(); it != prob.end(); ++it) 
            *it /= sum;
    }

    Factor operator*(const Factor& rhs) const {
        Factor result(prob * rhs.prob);
        result.normalize();
        return result;
    }

    ~Factor() {   }
public:
    array_nd<double> prob;  
};

class EntropyEstimator {
public:
    EntropyEstimator(const Dataset& dataset): dataset(dataset) { }

    double operator()(const array_1d<bool>& vars) {
        //if (!accumulate(vars.cbegin(), vars.cend(), false, logical_or<bool>()))
            //return 0.;
        Factor fact(dataset, vars);
        double result = 0.;
        for (auto& p: fact.prob) {
            result += p * log(p+1e-5);
        }
        return -result;
    }

    ~EntropyEstimator() { }
private:
    Dataset dataset;
};

class Score {
public:
    virtual double operator()(int x, const array_1d<bool>& pa) = 0;
};

class ScoreMI : public Score {
public:
    ScoreMI(const Dataset& dataset): dataset(dataset), entropy_estimator(dataset) { }

    virtual double operator()(int x, const array_1d<bool>& pa) {
        array_1d<bool> vars(pa);
        double H_Pa = entropy_estimator(vars); 
        vars(x) = true;
        double H_x_Pa = entropy_estimator(vars);
        for (int i = 0; i < vars.size(); i++)
            vars(i) = false;
        vars(x) = true;
        double H_x = entropy_estimator(vars);
        return H_Pa + H_x - H_x_Pa;
    }

    const Dataset& dataset;
    EntropyEstimator entropy_estimator;
};

class ScoreBIC : public Score {
public:
    ScoreBIC(const Dataset& dataset): dataset(dataset), mi(dataset) { }

    virtual double operator()(int x, const array_1d<bool>& pa) {
        double k = 1.;
        for (int i = 0; i < pa.size(); i++)
            if (pa(i)) k += dataset.n_values(i);
        k *= dataset.n_values(x);
        double n = dataset.rows();
        double l = n*mi(x, pa);
        return l - 0.5*log(n)*k;
    }

    const Dataset& dataset;
    ScoreMI mi;
};

class DiGraph {
public:
    DiGraph(int V) : adj({V, V}) {
        
    }

    const int V() const {
        return adj.shape(0);
    }

public:
    array_nd<bool> adj;
};

class LocalOperation {
public:
    LocalOperation(const DiGraph& graph, const Score& score) {
        
    }
private:

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
    vars(11) = true;
    EntropyEstimator ee(dataset);
    ScoreBIC mi(dataset);
    cout << mi(0, vars) << endl;
    return 0;
}

