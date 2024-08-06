#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <iostream>
#include <cmath>

namespace py = pybind11;
using Eigen::MatrixXd;
using Eigen::VectorXd;

std::pair<MatrixXd, MatrixXd> lanczosFastMult(MatrixXd w_mat, VectorXd deltas, int k) {
    int n = w_mat.cols();
    VectorXd v = VectorXd::Ones(n);

    // Call fastMult through Python
    py::object fastMultModule = py::module_::import("FastMult");
    py::object fastMult = fastMultModule.attr("fastMult");

    VectorXd degree = fastMult(w_mat.transpose(), deltas, VectorXd::Ones(n)).cast<VectorXd>();
    VectorXd inv_degree = degree.array().sqrt().inverse();

    MatrixXd Q = MatrixXd::Constant(n, k, std::nan(""));
    VectorXd q = v / v.norm();
    Q.col(0) = q;

    VectorXd d = VectorXd::Constant(k, std::nan(""));
    VectorXd od = VectorXd::Constant(k - 1, std::nan(""));

    for (int i = 0; i < k; ++i) {
        VectorXd z = degree.asDiagonal() * q - fastMult(w_mat.transpose(), deltas, q).cast<VectorXd>();

        d(i) = q.transpose() * z;

        // Reorthogonalization
        if (i > 0) {
            z -= Q.leftCols(i + 1) * (Q.leftCols(i + 1).transpose() * z);
            z -= Q.leftCols(i + 1) * (Q.leftCols(i + 1).transpose() * z);
        }

        if (i != k - 1) {
            od(i) = z.norm();
            q = z / od(i);
            Q.col(i + 1) = q;
        }
    }

    // Construct tridiagonal matrix T
    MatrixXd T = MatrixXd::Zero(k, k);
    T.diagonal() = d;
    T.diagonal(1) = od.head(k - 1);
    T.diagonal(-1) = od.head(k - 1);

    return {T, Q};
}

PYBIND11_MODULE(lanczosFastMultModule, m) {
    m.doc() = "pybind11 wrapper for the lanczosFastMult function"; // Optional module docstring
    m.def("lanczosFastMult", &lanczosFastMult, "A function that performs the Lanczos algorithm with full reorthogonalization using a custom matrix-vector product",
          py::arg("w_mat"), py::arg("deltas"), py::arg("k"));
}


