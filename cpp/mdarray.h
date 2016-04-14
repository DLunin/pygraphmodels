#ifndef _MDARRAY_H_
#define _MDARRAY_H_

#include <cassert>
#include <string>
#include <memory>
#include <algorithm>
#include <cstring>

#include <iostream>
#include <initializer_list>

using namespace std;

template <typename T>
class array_1d;

template <typename T>
ostream& operator<<(ostream& ostr, const array_1d<T>& arr);

template <typename T>
class array_1d {
public:
    array_1d(int n) : _ptr(static_cast<T*>(calloc(n, sizeof(T)))), _n(n), _is_view(false) {  }
    array_1d(initializer_list<T> init) : _ptr(static_cast<T*>(calloc(init.size(), sizeof(T)))), _n(init.size()), _is_view(false) {  
        auto it = init.begin(); 
        for (int i = 0; i < _n; ++i, ++it) 
            *const_cast<T*>(_ptr + i) = *it;
    }

    array_1d(const array_1d<T>& arr) : _ptr(static_cast<T*>(calloc(arr.size(), sizeof(T)))), _n(arr.size()), _is_view(false) {
        memcpy(_ptr, arr._ptr, _n * sizeof(T));
    }

    array_1d(T* ptr, int n) : _ptr(ptr), _n(n), _is_view(true) { }
    array_1d(const T* ptr, int n) : _ptr(ptr), _n(n), _is_view(true) { }

    const int nd() const {
        return 1;
    }

    const int size() const {
        return _n;
    }   

    const array_1d shape() const {
        return array_1d<int>({size()});
    }
    
    T& operator()(int idx) {
        return *(_ptr + idx);
    } 

    const T& operator()(int idx) const {
        return *(_ptr + idx);
    }

    T& operator[](const array_1d<int>& idx) {
        assert(idx.size() == 1);
        return operator()(idx(0));
    }

    const T& operator[](const array_1d<int>& idx) const {
        assert(idx.size() == 1);
        return operator()(idx(0));
    }

    array_1d<T> operator[](const array_1d<bool>& idx) const {
        int n_true = accumulate(idx.cbegin(), idx.cend(), 0);
        array_1d<T> result(n_true);
        int counter = 0;
        for (int i = 0; i < idx.size(); i++) {
            if (idx(i)) 
                result(counter++) = operator()(i);
        }
        return result;
    }

    T* begin() {
        return _ptr;
    }

    const T* cbegin() const {
        return _ptr;
    }

    T* end() {
        return _ptr + _n;
    }

    const T* cend() const {
        return _ptr + _n;
    }
    
    ~array_1d() {  
        if (!_is_view)
            free(_ptr);
    }
private:
    T *_ptr;
    const int _n;
    const bool _is_view;
};

template <typename T>
ostream& operator<<(ostream& ostr, const array_1d<T>& arr) {
    ostr << "array_1d([";
    for (int i = 0; i < arr.size() - 1; i++)
        ostr << arr(i) << ", ";
    if (arr.size() >= 1) 
        ostr << arr(arr.size() - 1);
    ostr << "])";
    return ostr;
}

template <typename T>
class array_nd {
public:
    array_nd(const array_1d<int>& shape) : _shape(shape), 
        _ptr(static_cast<T*>(calloc(accumulate(shape.cbegin(), shape.cend(), 1, multiplies<int>()), sizeof(T)))),
        _stride(shape.size()) {
    
        _init_stride();  
    }
    
    array_nd(initializer_list<int> shape) : _shape(shape), 
        _ptr(static_cast<T*>(calloc(accumulate(shape.begin(), shape.end(), 1, multiplies<int>()), sizeof(T)))), 
        _stride(shape.size()) {

        _init_stride();  
    }

    array_nd(const array_nd<T>& arr) : _shape(arr._shape),
        _ptr(static_cast<T*>(calloc(accumulate(arr._shape.cbegin(), arr._shape.cend(), 1, multiplies<int>()), sizeof(T)))),
        _stride(arr._stride)  {
        
        memcpy(_ptr, arr._ptr, size() * sizeof(T));
    }

    const int nd() const {
        return shape.size();
    }

    const array_1d<int> shape() const {
        return _shape;
    }

    const int shape(int dim) const {
        return _shape(dim);
    }

    const int size() const {
        return accumulate(_shape.cbegin(), _shape.cend(), 1, multiplies<int>());
    }

    T& operator[](const array_1d<int>& idx) {
#ifdef DEBUG
        assert(inner_product(idx.cbegin(), idx.cend(), _stride.cbegin(), 0) < size());
        assert(inner_product(idx.cbegin(), idx.cend(), _stride.cbegin(), 0) >= 0);
        assert(idx.size() == _stride.size());
#endif
        return *(_ptr + inner_product(idx.cbegin(), idx.cend(), _stride.cbegin(), 0));
    }

    const T& operator[](const array_1d<int>& idx) const {
#ifdef DEBUG
        assert(inner_product(idx.cbegin(), idx.cend(), _stride.cbegin(), 0) < size());
        assert(inner_product(idx.cbegin(), idx.cend(), _stride.cbegin(), 0) >= 0);
        assert(idx.size() == _stride.size());
#endif
        return *(_ptr + inner_product(idx.cbegin(), idx.cend(), _stride.cbegin(), 0));
    }

    const T* ptr_to(const array_1d<int>& idx) const {
        return (_ptr + inner_product(idx.cbegin(), idx.cend(), _stride.cbegin(), 0));
    }

    T* ptr_to(const array_1d<int>& idx) {
        return (_ptr + inner_product(idx.cbegin(), idx.cend(), _stride.cbegin(), 0));
    }

    T* begin() {
        return _ptr;
    }

    const T* cbegin() const {
        return _ptr;
    }

    T* end() {
        return _ptr + size();
    }

    const T* cend() const {
        return _ptr + size();
    }

    ~array_nd() {  
        free(_ptr); 
    }

private:
    void _init_stride() const {
        const int n = _stride.size();
        const_cast<int&>(_stride(n-1)) = 1;
        for (int i = 1; i < _shape.size(); i++)
            const_cast<int&>(_stride(n-i-1)) = _stride(n-i) * shape(i);
    }

private:
    T *_ptr;
    const array_1d<int> _shape;
    const array_1d<int> _stride;

};

#endif
