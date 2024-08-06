#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <cstddef> // For std::ptrdiff_t

namespace py = pybind11;

py::array_t<double> fastMult(py::array_t<double> W_T, py::array_t<double> deltas, py::array_t<double> v) {
    auto r = v.request(), d = deltas.request(), w = W_T.request();

    if (r.ndim != 2 || r.shape[1] != 1)
        throw std::runtime_error("Input vector v must be of shape (n, 1)");

    if (d.ndim != 1)
        throw std::runtime_error("Deltas must be a one-dimensional array of scalars");

    if (w.ndim != 2)
        throw std::runtime_error("Input matrix W_T must be two-dimensional");

    size_t n = w.shape[0];
    size_t m = w.shape[1];

    // Create a zero-initialized array properly
    auto result = py::array_t<double>(r.shape);
    auto result_buf = result.mutable_data(); // Get a pointer to the result data buffer
    std::fill(result_buf, result_buf + n, 0.0); // Initialize all elements to zero

    auto buf_result = result.mutable_unchecked<2>();  // buffer for result array
    auto buf_v = v.unchecked<2>();
    auto buf_deltas = deltas.unchecked<1>();
    auto buf_W_T = W_T.unchecked<2>();

    std::vector<double> col_sum(m, 0.0);

    #pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < m; i++) {
        double dot_product = 0.0;
        for (std::ptrdiff_t k = 0; k < n; k++) {
            dot_product += buf_W_T(k, i) * buf_v(k, 0);
        }
        col_sum[i] = dot_product;
    }

    #pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < m; i++) {
        for (std::ptrdiff_t j = 0; j < n; j++) {
            double temp_result = (col_sum[i] * buf_W_T(j, i)) / buf_deltas(i);
            buf_result(j, 0) += temp_result;
        }
    }

    return result;
}

PYBIND11_MODULE(fastMultModule, m) {
    m.doc() = "pybind11 wrapper for the fastMult function";
    m.def("fastMult", &fastMult, "A function that multiplies a matrix G_k (based on the ACA components) and a vector v efficiently",
          py::arg("W_T"), py::arg("deltas"), py::arg("v"));
}
