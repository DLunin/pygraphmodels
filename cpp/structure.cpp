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
#include <map>
#include "csv.h"
#include "mdarray.h"

using namespace std;

class Dataset;
class DatsetView;

class DiGraph {
public:
    DiGraph(int V) : adj({V, V}) {
        
    }

    const int V() const {
        return adj.shape(0);
    }

    const array_1d<bool> pa(int x) const {
        array_1d<bool> result(V());
        auto ptr = adj.ptr_to({x});
        std::copy(ptr, ptr + V(), result.begin());
        return result;
    }

    const bool has_edge(int from, int to) const {
        return adj[{to, from}];
    }

    void add_edge(int from, int to) {
        adj[{to, from}] = true;
    }
    
    void remove_edge(int from, int to) {
        adj[{to, from}] = false;
    }

    const bool _dfs(int v, int* colors) const {
        if (colors[v] == 1) 
            return false;
        if (colors[v] == 2)
            return true;
        colors[v] = 1;
        for (int i = 0; i < V(); i++) {
            if (has_edge(v, i)) {
                if (!_dfs(i, colors))
                    return false;
            }
        }
        colors[v] = 2;
        return true;
    }

    const bool is_acyclic() const {
        int colors[V()];
        for (int i = 0; i < V(); i++)
            colors[i] = 0;

        for (int i = 0; i < V(); i++) {
            if (colors[i] == 0)
                if (!_dfs(i, colors))
                    return false;
        }
        return true;
    }

public:
    array_nd<bool> adj;
};

ostream& operator<<(ostream& ostr, const DiGraph& g) {
    ostr << "Graph has " << g.V() << " vertices." << endl;
    for (int i = 0; i < g.V(); i++) {
        for (int j = 0; j < g.V(); j++) 
            if (g.has_edge(i, j)) {
                ostr << i << " -> " << j << endl;
            }
    }
    ostr << "-----" << endl;
    return ostr;
}


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

    double operator()(const array_1d<bool>& vars) const {
        if (!accumulate(vars.cbegin(), vars.cend(), false, logical_or<bool>()))
            return 0.;
        auto cache_pos = cache.find(vars);
        if (cache_pos != cache.end()) 
            return cache_pos->second;
        Factor fact(dataset, vars);
        double result = 0.;
        for (auto& p: fact.prob) {
            result += p * log(p+1e-5);
        }
        return cache[vars] = -result;
    }

    ~EntropyEstimator() { }
private:
    Dataset dataset;
    mutable map<array_1d<bool>, double> cache;
};

class Score {
public:
    virtual double operator()(int x, const array_1d<bool>& pa) const = 0;
    virtual double total(const DiGraph& graph) const {
        double result = 0;
        for (int i = 0; i < graph.V(); i++) {
            result += operator()(i, graph.pa(i));
        }
        return result;
    }
};

class ScoreMI : public Score {
public:
    ScoreMI(const Dataset& dataset): dataset(dataset), entropy_estimator(dataset) { }

    virtual double operator()(int x, const array_1d<bool>& pa) const {
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

    virtual double operator()(int x, const array_1d<bool>& pa) const {
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
class LocalOperation {
public:
    virtual 
    const double score() = 0;

    virtual 
    const bool apply() = 0;
};

class AddEdge : public LocalOperation {
public:
    AddEdge(DiGraph& graph, const Score& score, int from, int to) : _graph(graph), _score(score), from(from), to(to) {
         
    }

    virtual
    const double score() override {
        if (_graph.has_edge(from, to))
            return 0;

        _graph.add_edge(from, to);
        if (!_graph.is_acyclic()) {
            _graph.remove_edge(from, to);
            return 0;
        }
        _graph.remove_edge(from, to);

        auto pa = _graph.pa(to);
        double score = -_score(to, pa);
        pa(from) = true;
        score += _score(to, pa);
        return score;
    }

    virtual 
    const bool apply() override {
        if (_graph.has_edge(from, to))
            return false;
        _graph.add_edge(from, to);
        if (!_graph.is_acyclic()) {
            _graph.remove_edge(from, to);
            return false;   
        }
        return true;
    }

private:
    int from, to;
    DiGraph& _graph;
    const Score& _score;
};

class RemoveEdge : public LocalOperation {
public:
    RemoveEdge(DiGraph& graph, const Score& score, int from, int to) : _graph(graph), _score(score), from(from), to(to) {
         
    }

    virtual 
    const double score() override {
        if (!_graph.has_edge(from, to))
            return 0;
        array_1d<bool> pa = _graph.pa(to);
        double score = -_score(to, pa);
        pa(from) = false;
        score += _score(to, pa);
        return score;
    }

    virtual 
    const bool apply() override {
        if (!_graph.has_edge(from, to))
            return false;
        _graph.remove_edge(from, to);
        return true;
    }

private:
    int from, to;
    DiGraph& _graph;
    const Score& _score;
};

class ReverseEdge : public LocalOperation {
public:
    ReverseEdge(DiGraph& graph, const Score& score, int from, int to) : _graph(graph), _score(score), from(from), to(to) {
         
    }

    virtual 
    const double score() override {
        if (!_graph.has_edge(from, to))
            return 0;

        _graph.remove_edge(from, to);
        _graph.add_edge(to, from);
        if (!_graph.is_acyclic()) {
            _graph.remove_edge(to, from);
            _graph.add_edge(from, to);
            return 0;
        }
        _graph.remove_edge(to, from);
        _graph.add_edge(from, to);

        auto pa = _graph.pa(to);
        double score = -_score(to, pa);
        pa(from) = false;
        score += _score(to, pa);

        pa = _graph.pa(from);
        score -= _score(from, pa);
        pa(to) = true;
        score += _score(from, pa);
        return score;
    }

    virtual
    const bool apply() override {
        if (!_graph.has_edge(from, to))
            return false;
        _graph.remove_edge(from, to);
        _graph.add_edge(to, from);
        if (!_graph.is_acyclic()) {
            _graph.remove_edge(to, from);
            _graph.add_edge(from, to);
            return false;   
        }
        return true;
    }

private:
    int from, to;
    DiGraph& _graph;
    const Score& _score;
};

template <typename T, typename KeyOp, typename TKey>
class Heap {
public:
    template <typename Container>
    Heap(const Container& container, int max_size, KeyOp key_f): _key_f(key_f), _arr(max_size), _keys(max_size), _size(container.cend() - container.cbegin()) {
        std::copy(container.cbegin(), container.cend(), _arr.begin());
        for (int i = 0; i < _keys.size(); i++)
            _keys(i) = key_f(_arr(i));
        for (int i = min(_size - 1, _size / 2 + 1); i >= 0; i--)
            _heapify(i);
    }

    const int _left(int i) const {
        return 2*i + 1;
    }

    const int _right(int i) const {
        return 2*i + 2;
    }

    const int _parent(int i) const {
        return (i - 1) / 2;
    }
    
    void _heapify(int i) {
        if (i >= _size) return;
        int largest = i;
        if (_left(i) < _size && _keys(_left(i)) > _keys(largest)) {
            largest = _left(i);
        }
        if (_right(i) < _size && _keys(_right(i)) > _keys(largest)) {
            largest = _right(i);
        }
        if (largest != i) {
            swap(_arr(i), _arr(largest));
            swap(_keys(i), _keys(largest));
            _heapify(largest);
        }
    }

    T& top() {
        return _arr(0);
    }

    void key_changed(int i) {
        TKey new_key = _key_f(_arr(i));
        TKey old_key = _keys(i);
        _keys(i) = new_key;
        if (new_key < old_key) {
            _key_decreased(i);
        } 
        else {
            _key_increased(i);
        }
    }

    void _key_decreased(int i) {
        _heapify(i);
    }

    void _key_increased(int i) {
        while (i > 0 && _keys(i) > _keys(_parent(i))) {
            swap(_keys(i), _keys(_parent(i)));
            swap(_arr(i), _arr(_parent(i)));
            i = _parent(i);
        }
    }

private:
    int _size;
    KeyOp _key_f;
    array_1d<T> _arr;
    array_1d<TKey> _keys;
};

class GreedySearch {
public:
    GreedySearch(DiGraph& graph, const Score& score): graph(graph), score(score) {
        for (int i = 0; i < graph.V(); i++) { 
            for (int j = 0; j < graph.V(); j++) {
                if (i == j) continue;
                operations.push_back(make_shared<AddEdge>(graph, score, i, j));
                operations.push_back(make_shared<RemoveEdge>(graph, score, i, j));
                operations.push_back(make_shared<ReverseEdge>(graph, score, i, j));
            }
        }
    }

    bool iteration() {
        auto op = *std::max_element(operations.rbegin(), operations.rend(), [](shared_ptr<LocalOperation> op1, shared_ptr<LocalOperation> op2){ return op1->score() < op2->score(); });
        if (op->score() < 0) 
            return true;
        if (op->apply())
            return false;
        return true;
    }

    const DiGraph& operator()() {
        int counter = 1;
        cout << "hello" << endl;
        while (!iteration()) {
            cout << "iteration " << counter++ << ": " << score.total(graph) << endl;
        }
        return graph;
    }

private:
    DiGraph& graph;
    const Score& score;
    vector<shared_ptr<LocalOperation>> operations;
};

template <typename T>
double call_score(const T& x) {
    return x->score();
}

class HeapGreedySearch {
public:
    HeapGreedySearch(DiGraph& graph, const Score& score): graph(graph), score(score), 
    operations([&graph, &score]() {
        vector<shared_ptr<LocalOperation>> ops;
        for (int i = 0; i < graph.V(); i++) { 
            for (int j = 0; j < graph.V(); j++) {
                if (i == j) continue;
                ops.push_back(make_shared<AddEdge>(graph, score, i, j));
                ops.push_back(make_shared<RemoveEdge>(graph, score, i, j));
                ops.push_back(make_shared<ReverseEdge>(graph, score, i, j));
            }
        }
        return ops;
    }(), 3 * graph.V() * (graph.V() - 1), &call_score) {

    }

    bool iteration() {
        return true;
    }

    const DiGraph& operator()() {
        int counter = 1;
        cout << "hello" << endl;
        while (!iteration()) {
            cout << "iteration " << counter++ << ": " << score.total(graph) << endl;
        }
        return graph;
    }

private:
    DiGraph& graph;
    const Score& score;
    Heap<shared_ptr<LocalOperation>, double (*)(const shared_ptr<LocalOperation>&), double> operations;
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
        cout << i << ". " << dataset.headers[i] << ": " << dataset.n_values(i) << endl;
    }

    array_1d<bool> vars(dataset.cols());
    vars(11) = true;
    EntropyEstimator ee(dataset);
    ScoreBIC mi(dataset);
    cout << mi(0, vars) << endl;
    cout << "=====" << endl;

    DiGraph g(dataset.cols());
    GreedySearch gs(g, mi);
    gs();
    cout << "success" << endl;
    return 0;
}
