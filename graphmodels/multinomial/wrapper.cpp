#include <boost/python.hpp>
#include <iostream>
#include <cassert>

#include "ndarray.h"
#include <cmath>

using namespace std;
namespace py = boost::python;

void gen_multinomial(PyObject *arr, PyObject *acc_prob_arr) {
    std::default_random_engine re(rand());
    std::uniform_real_distribution<double> unif(0, 1);

    ndarray<double> result(arr), acc_prob(acc_prob_arr);
    assert(result.nd() == 1);
    assert(acc_prob.nd() == 2);
    assert(acc_prob.shape(0) == result.shape(0));
    for (int i = 0; i < result.shape(0); i++) {
        double current = unif(re);
        int j = 0;
        for (; j < acc_prob.shape(1) && current > acc_prob(i, j); j++) ;
        result(i) = j;
    }
}

BOOST_PYTHON_MODULE(multinomial_cpp)
{
    py::def("gen_multinomial", &gen_multinomial);
}
