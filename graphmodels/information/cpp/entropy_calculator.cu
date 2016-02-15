#ifndef _ENTROPY_CALCULATOR_H
#define _ENTROPY_CALCULATOR_H

#include <type_traits>
#include <memory>
#include <iostream>
#include <utility>
#include <vector>
#include <algorithm>
#include <cstdlib>
#include <random>

#include <Python.h>
#include <boost/python.hpp>
#include <boost/function.hpp>
#include <iostream>
#include <functional>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "ndarray.h"

#include <armadillo>

void gpu_blas_mmul(const double *D, const double *E, double *F, const int rowD, const int colD, const int colE) {
    // All matrices are supposed to be in normal, C-style layout

	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Do the actual multiplication
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, colE, rowD, colD, alpha, E, colE, D, colD, beta, F, colE);

	// Destroy the handle
	cublasDestroy(handle);
}

using namespace std;
using thrust::device_vector;
using thrust::host_vector;

using arma::mat;

typedef device_vector<double> dvec;
typedef host_vector<double> hvec;
typedef shared_ptr<device_vector<double>> dvec_ptr;
typedef device_vector<double>::iterator dvec_iter;

class pseudo_dvec {
public:
    pseudo_dvec(dvec_ptr ptr, dvec_iter begin, dvec_iter end) : storage(ptr), _begin(begin), _end(end) { }

    double operator[](int idx) { 
        return *(_begin + idx);
    }

    dvec_iter begin() { 
        return _begin;
    }

    dvec_iter end() {
        return _end;
    }

    int size() { 
        return _end - _begin; 
    }

private:
    dvec_iter _begin, _end;
    dvec_ptr storage;
};

typedef shared_ptr<pseudo_dvec> pseudo_dvec_ptr;

template <typename dvec_ptr>
double mean(dvec_ptr v) {
    double n = v->end() - v->begin();
    return thrust::reduce(v->begin(), v->end()) / n;
}


template <typename dvec_ptr>
double covariance(dvec_ptr v1, dvec_ptr v2) {
    // VECTOR MEAN MUST BE EQUAL TO ZERO!
    double n = v1->end() - v1->begin();
    return thrust::inner_product(v1->begin(), v1->end(), v2->begin(), static_cast<double>(0.0)) / (n - 1);
}

void debug_print(const arma::mat& m) {
    for (int i = 0; i < m.n_rows; i++) {
        for (int j = 0; j < m.n_cols; j++) {
            cout << m(i, j) << " ";
        }
        cout << endl;
    }
}

void print_data(const dvec& v, int n_vars, int n_samples) {
    for (int i = 0; i < n_vars; i++) {
        for (int j = 0; j < n_samples; j++) 
            cout << v[n_samples*i + j] << " ";
        cout << endl;
    }
}

class DataManager {
public:
    DataManager(PyObject *arr_obj) : arr(arr_obj), n_vars(arr.shape(0)), n_samples(arr.shape(1)), loaded(new dvec(n_vars * n_samples)) { 
        const double *ptr = arr.plain_mem();
        if (ptr) {
            // fast copy
            cout << "fast copy" << endl;
            thrust::copy(arr.plain_mem(), arr.plain_mem() + n_vars*n_samples, loaded->begin());
        }
        else {
            // slow copy
            cout << "slow copy" << endl;
            host_vector<double> temp(n_vars * n_samples);
            for (int i = 0; i < n_vars; i++) {
                for (int j = 0; j < n_samples; j++)
                    temp[i*n_samples + j] = arr(i, j);
            }
            thrust::copy(temp.begin(), temp.end(), loaded->begin());
        }
        normalize_mean();
        cov = calc_cov();
    }
    
    pseudo_dvec_ptr get(int idx) const {
        return make_shared<pseudo_dvec>(loaded, loaded->begin() + idx*n_samples, loaded->begin() + (idx+1)*n_samples); 
    }

    template <typename TIter>
    vector<pseudo_dvec_ptr> operator()(TIter vbegin, TIter vend) const {
        int n_query = vend - vbegin;
        vector<pseudo_dvec_ptr> result(n_query, nullptr);
        dvec_ptr res_storage = make_shared<dvec>(n_vars * n_samples);
        arma::mat tr, itr;
        tie(tr, itr) = get_transforms(vbegin, vend); 
        dvec dev_tr = shaded(tr, vbegin, vend);
        gpu_blas_mmul(thrust::raw_pointer_cast(&dev_tr[0]), 
                thrust::raw_pointer_cast(&(*loaded)[0]),
                thrust::raw_pointer_cast(&(*res_storage)[0]),
                n_query, n_vars, n_samples);
        for (int i = 0; i < n_query; i++, vbegin++) {
            dvec_iter cur_begin = res_storage->begin() + i*n_samples;
            dvec_iter cur_end = res_storage->begin() + (i + 1)*n_samples;
            result[i] = make_shared<pseudo_dvec>(res_storage, cur_begin, cur_end);
        }
        return result; 
    }

    template <typename TIter>
    vector<double> get_means(TIter begin, TIter end) const {
        vector<double> result;
        for (; begin != end; begin++)
            result.push_back(means[*begin]);
        return result;
    }

    arma::mat calc_cov() const {
        arma::mat result(n_vars, n_vars);
        for (int i = 0; i < n_vars; i++) {
            for (int j = 0; j <= i; j++) {
                result(i, j) = result(j, i) = covariance(get(i), get(j));
            }
        }
        return result;
    }

    arma::mat get_cov() const {
        return cov;
    }

    template <typename TIter>
    arma::mat get_cov(TIter begin, TIter end) const {
        vector<int> temp(begin, end);
        arma::mat result(temp.size(), temp.size());
        for (int i = 0; i < temp.size(); i++) {
            for (int j = 0; j <= i; j++) {
                result(i, j) = result(j, i) = cov(temp[i], temp[j]);
            }
        }
        return result;
    }

    template <typename TIter>
    dvec shaded(const mat& m, TIter begin, TIter end) const {
        assert(m.n_rows == m.n_cols);
        assert(m.n_rows == (end - begin));
        int n_query = m.n_rows;

        
        mat temp(n_query, n_vars, arma::fill::zeros);
        for (int i = 0; i < n_query; i++) {
            temp(i, *begin++) = 1;
        }
        temp = m * temp;

        host_vector<double> result(n_query * n_vars, 0);
        for (int i = 0; i < n_query; i++) 
            for (int j = 0; j < n_vars; j++) 
                result[i*n_vars + j] = temp(i, j);
        return dvec(result.begin(), result.end());
    }

    template <typename... TArgs>
    tuple<arma::mat, arma::mat> get_transforms(TArgs... args) const {
        arma::mat result = arma::chol(get_cov(args...), "lower");
        return make_tuple(arma::inv(result), result);
    }

    vector<pseudo_dvec_ptr> operator()() const {
        vector<int> range(n_vars);
        for (int i = 0; i < n_vars; i++)
            range[i] = i;
        return operator()(range.begin(), range.end()); 
    }


private:
    void normalize_mean() {
        means.resize(n_vars);
        for (int i = 0; i < n_vars; i++) {
            auto current = get(i);
            means[i] = mean(current);
            thrust::transform(current->begin(), current->end(), thrust::constant_iterator<double>(means[i]), current->begin(), thrust::minus<double>());
        }
    }

private:
    const ndarray<double> arr;
public:
    const int n_vars, n_samples;
    mutable vector<double> means;
    arma::mat cov;
private:
    mutable dvec_ptr loaded;
};

struct KDE_transform {
    const double x;
    KDE_transform(double x) : x(x) { }

    __host__ __device__ 
    double operator()(const double& p, const double& m) {
        return p + (m - x) * (m - x);
    }
};

struct KDE_map {
    const double sigma2;
    KDE_map(double sigma2) : sigma2(sigma2) { }

    __host__ __device__ 
    double operator()(const double& x) {
        return exp(-(1. / (2 * sigma2))*x);
    }
};

template <typename dvec_ptr = pseudo_dvec>
class KDE_t {
public:
    KDE_t(const vector<dvec_ptr>& data, const vector<double>& means, const mat& tm, const mat& itm, double sigma2_=0.0) : 
        data(data), 
        n_vars(data.size()), 
        n_samples(data[0]->size()), 
        sigma2(sigma2_),
        generator(),
        distribution(0, 1),
        means(means),
        tm(tm), itm(itm), volume(arma::det(tm))
    { 
        auto v1 = data[0];
        if (sigma2 == 0.0) {
            sigma2 = pow(n_samples, -1. / (n_vars + 4));
        }
    }

    template <typename TIter>
    double operator()(TIter xbegin) {
        mat x(n_vars, 1);
        for (int i = 0; i < n_vars; i++, xbegin++)
            x(i, 0) = *xbegin - means[i];
        x = tm * x;
        device_vector<double> a(n_samples);
        for (int i = 0; i < n_vars; i++) {      
            thrust::transform(a.begin(), a.end(), data[i]->begin(), a.begin(), KDE_transform(x(i, 0))); 
        }
        thrust::transform(a.begin(), a.end(), a.begin(), KDE_map(sigma2));
        double res = thrust::reduce(a.begin(), a.end(), (double)0.0, thrust::plus<double>());
        return (res / (n_samples * pow(2*M_PI*sigma2, n_vars / 2.))) * volume;
    }

    template <typename TIter>
    void sample(TIter result) {
        mat x(n_vars, 1);
        int kernel_i = rand() % n_samples;
        for (int i = 0; i < n_vars; i++) {
            double noise = distribution(generator) * sqrt(sigma2); 
            x(i, 0) = noise + (*data[i])[kernel_i];
        }
        x = itm * x;
        for (int i = 0; i < n_vars; i++, result++)
            *result = x(i, 0) + means[i];
    }


private:
    vector<dvec_ptr> data;
public:
    const int n_vars;
    const int n_samples;
    double sigma2;
    double volume;
    default_random_engine generator;
    normal_distribution<double> distribution;
    vector<double> means;
    mat tm, itm;
};

typedef KDE_t<pseudo_dvec_ptr> KDE;

template <typename TIter>
KDE make_kde(const DataManager& dm, TIter begin, TIter end) {
    auto data = dm(begin, end);
    vector<double> means = dm.get_means(begin, end);
    mat tm, itm;
    tie(tm, itm) = dm.get_transforms(begin, end);
    return KDE(data, means, tm, itm);
}

KDE make_kde(const DataManager& dm) {
    vector<int> range(dm.n_vars);
    for (int i = 0; i < range.size(); i++)
        range[i] = i;
    return make_kde(dm, range.begin(), range.end());
}

class EntropyCalculator {
public:
    EntropyCalculator(PyObject *arr_obj) : dm(arr_obj) { }

    template <typename TIterX, typename TIterY>
    double operator()(TIterX xbegin, TIterX xend, TIterY ybegin, TIterY yend, int n_iter=1000) {
        int nx = xend - xbegin, ny = yend - ybegin;
        vector<double> xy(nx + ny, 0);

        vector<int> xidx(nx), yidx(ny), xyidx(nx + ny);
        copy(xbegin, xend, xidx.begin());
        copy(xbegin, xend, xyidx.begin());
        copy(ybegin, yend, yidx.begin());
        copy(ybegin, yend, xyidx.begin() + nx);

        KDE p_xy = make_kde(dm, xyidx.begin(), xyidx.end());
        KDE p_x = make_kde(dm, xidx.begin(), xidx.end());
        KDE p_y = make_kde(dm, yidx.begin(), yidx.end());

        long double result = 0;
        for (int i = 0; i < n_iter; i++) {
            p_xy.sample(xy.begin());
            auto temp = log(p_xy(xy.begin())) - log(p_x(xy.begin())) - log(p_y(xy.begin() + nx));
            result += temp;
        } 

        result /= n_iter;
        return result;
    }

    ~EntropyCalculator() { }    
private:   
    DataManager dm;
};

#endif
