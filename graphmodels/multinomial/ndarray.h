#ifndef _NDARRAY_H_
#define _NDARRAY_H_

#include <cassert>
#include <string>
#include <Python.h>
#include <boost/python.hpp>
#include <numpy/ndarrayobject.h>

#include <iostream>
using namespace std;

namespace py = boost::python;

template <typename T>
struct ptrjump_iter {
    typedef T value_type;
    typedef int difference_type;
    typedef T* pointer;
    typedef T& reference;
    typedef forward_iterator_tag iterator_category;

    ptrjump_iter(T* ptr, int jump) : ptr(ptr), jump(jump) {   }
    
    ptrjump_iter operator++(int) {
        auto saved = ptr;
        ptr += jump;
        return ptrjump_iter(saved, jump);
    }

    ptrjump_iter& operator++() {
        ptr += jump;
        return *this;
    }

    T& operator*() {
        return *ptr;
    }

    int operator-(const ptrjump_iter& rhs) const {
        assert(jump == rhs.jump);
        return (ptr - rhs.ptr) / jump;
    }

    bool operator==(const ptrjump_iter& rhs) const {
        return rhs.ptr == ptr && rhs.jump == jump;
    }

    bool operator!=(const ptrjump_iter& rhs) const {
        return !operator==(rhs);
    }

    T* ptr;
    int jump;
};

template <typename T>
class ndarray {
public: 
    ndarray(PyObject *obj) : _obj(obj), custom_strides(false) {
        Py_INCREF(obj);
        PyObject *s_temp = PyUnicode_FromString("__array_struct__");
        _array_struct = PyObject_GetAttr(obj, s_temp);
        _arr = static_cast<PyArrayInterface*>(PyCapsule_GetPointer(_array_struct, PyCapsule_GetName(_array_struct))); 

        assert(_arr->two == 2);
        if (!_arr->strides) {
            _arr->strides = new npy_intp[1];
            _arr->strides[0] = sizeof(T);
            custom_strides = true;
        }
        assert(_arr->itemsize == sizeof(T));

        _data = reinterpret_cast<char*>(_arr->data);
    }

    ndarray(const ndarray& arr) : _arr(arr._arr), _array_struct(arr._array_struct), _obj(arr._obj), _data(arr._data), custom_strides(custom_strides) {
        Py_INCREF(_obj);
    }

    int nd() const { return _arr->nd; }
    int shape(int i) const { 
        return *(_arr->shape + i);
    }

    const T* plain_mem() const {
        cout << "plain_mem()" << endl;
        if (_arr->strides[nd() - 1] != sizeof(T)) {
            cout << _arr->strides[nd() - 1] << " != " << sizeof(T) << endl;
            return nullptr;
        }
        for (int i = nd() - 2; i >= 0; i--) {
            if (_arr->strides[i] != _arr->strides[i+1] * shape(i+1)) {
                cout << "failed at: " << _arr->strides[i] << " != " << _arr->strides[i+1] << " * " << shape(i + 1) << endl;
                return nullptr;
            }
        }
        return reinterpret_cast<const T*>(_data);
    }

    template <typename... TArg>
    int _get_offset(decltype(PyArrayInterface::strides) stride, int i, TArg... args) const {
        return (*stride) * i + _get_offset(stride + 1, args...);
    }

    int _get_offset(decltype(PyArrayInterface::strides) stride, int i) const {
        return (*stride) * i;
    }

    ptrjump_iter<T> begin() const {
        assert(nd() == 1);
        return ptrjump_iter<T>(reinterpret_cast<T*>(_data), _arr->strides[0] / sizeof(T));
    }
    ptrjump_iter<T> end() const {
        assert(nd() == 1);
        return ptrjump_iter<T>(reinterpret_cast<T*>(_data + _arr->strides[0] * _arr->shape[0]), _arr->strides[0] / sizeof(T));
    }

    template <typename... TArg>
    T operator()(TArg... args) const {
        return *reinterpret_cast<T*>(_data + _get_offset(_arr->strides, args...));
    }

    template <typename... TArg>
    T& operator()(TArg... args)  {
        return *reinterpret_cast<T*>(_data + _get_offset(_arr->strides, args...));
    }

    ~ndarray() {
        Py_DECREF(_array_struct);
        Py_DECREF(_obj);
    }
private:
    bool custom_strides;
    PyObject *_array_struct, *_obj;
    PyArrayInterface *_arr; 
    char *_data;
};  

template <typename T>
ostream& operator<<(ostream& ostr, const ndarray<T>& arr) {
    assert(arr.nd() <= 2);
    if (arr.nd() == 1) {
        cout << "shape: (" << arr.shape(0) << ")" << endl;
        for (int i = 0; i < arr.shape(0); i++) {
            ostr << arr(i) << " ";
        }
        ostr << endl;
    }
    else if (arr.nd() == 2) {
        cout << "shape: (" << arr.shape(0) << ", " << arr.shape(1) << ")" << endl;
        for (int i = 0; i < arr.shape(0); i++) {
            for (int j = 0; j < arr.shape(1); j++) 
                ostr << arr(i, j) << " ";
            ostr << endl;
        }
    }
    return ostr;
}

#endif
