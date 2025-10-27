#include "../pybind11/include/pybind11/pybind11.h"
#include "../pybind11/include/pybind11/stl.h"
#include "lab3.cpp"

namespace py = pybind11;
using lab3::Matrix;

PYBIND11_MODULE(matrix_module, m) {
    py::class_<Matrix<int>>(m, "Matrix")
        .def(py::init<size_t, size_t>())
        .def("get", [](const Matrix<int> &m, int i, int j) { return m(i, j); })
        .def("set", [](Matrix<int> &m, int i, int j, int val) { m(i, j) = val; })
        .def("rows", &Matrix<int>::Rows)
        .def("cols", &Matrix<int>::Cols)
        .def("__repr__", [](const Matrix<int> &m) {
            return "Matrix<" + std::to_string(m.Rows()) + "x" + std::to_string(m.Cols()) + ">";
        })
        .def("__str__", [](const Matrix<int> &mat) {
            std::string result;
            for (size_t i = 0; i < mat.Rows(); ++i) {
                for (size_t j = 0; j < mat.Cols(); ++j) {
                    result += std::to_string(mat(i, j)) + " ";
                }
                result += "\n";
            }
            return result;
        });
    
    m.def("create_4x4_from_16x16", [](const Matrix<int> &src) {
        Matrix<int> result(4, 4);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                int sum = 0;
                for (int x = 0; x < 4; x++) {
                    for (int y = 0; y < 4; y++) {
                        sum += src(i*4 + x, j*4 + y);
                    }
                }
                result(i, j) = sum;
            }
        }
        return result;
    });
}