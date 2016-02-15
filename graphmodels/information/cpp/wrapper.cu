#include <boost/python.hpp>
#include <iostream>
#include <cassert>

#include "ndarray.h"
#include "entropy_calculator.cu"
#include <cmath>

using namespace std;
namespace py = boost::python;

struct py_KDE {
    py_KDE(PyObject *obj, double sigma2 = 0.0) : dm(obj), kde(make_kde(dm)) { 
         
    }

    double pdf(PyObject* x) {
        ndarray<double> arr(x);
        return kde(arr.begin());
    }

    void sample(PyObject* aptr) {
        ndarray<double> arr(aptr);
        assert(arr.nd() == 2);
        vector<double> temp(arr.shape(0));
        for (int i = 0; i < arr.shape(1); i++) {
            kde.sample(temp.begin());
            for (int j = 0; j < arr.shape(0); j++)
                arr(j, i) = temp[j];
        }
    }


    DataManager dm;
    KDE kde;
};

struct py_EntropyCalculator {
    py_EntropyCalculator(PyObject *data) : calc(data) { }

    double mi(PyObject *x, PyObject *y, int n_iter) {
        ndarray<int> xarr(x), yarr(y);
        return calc(xarr.begin(), xarr.end(), yarr.begin(), yarr.end(), n_iter);
    }

    EntropyCalculator calc;
};

BOOST_PYTHON_MODULE(entcalc)
{
    py::class_<py_KDE>("KDE", py::init<PyObject*, double>())
        .def("pdf", &py_KDE::pdf)
        .def("sample", &py_KDE::sample);

    py::class_<py_EntropyCalculator>("EntCalc", py::init<PyObject*>())
        .def("mi", &py_EntropyCalculator::mi);
}
