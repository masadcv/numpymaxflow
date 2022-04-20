// Copyright (c) 2022, Muhammad Asad (masadcv@gmail.com)
// All rights reserved.

// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:

// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.

// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.

// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.

// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <vector>
#include <cmath>
#include <iostream>
#include <limits>
#include <Python.h>
#include <numpy/arrayobject.h>

// int get_pyarray_object_dim(PyArrayObject *data)
// {
//     return PyArray_NDIM(data);
// }

// void print_shape(PyArrayObject *data)
// {
//     int num_dims = get_pyarray_object_dim(data);
//     npy_intp *data_shape = PyArray_DIMS(data);
//     std::cout << "Shape: (";
//     for (int dim = 0; dim < num_dims; dim++)
//     {
//         std::cout << data_shape[dim];
//         if (dim != num_dims - 1)
//         {
//             std::cout << ", ";
//         }
//         else
//         {
//             std::cout << ")" << std::endl;
//         }
//     }
// }

// void check_spatial_shape_match(PyArrayObject *in1, PyArrayObject *in2, const int &dims)
// {

//     if (get_pyarray_object_dim(in1) != get_pyarray_object_dim(in2))
//     {
//         throw std::runtime_error("dimensions of input tensors do not match "
//             + std::to_string(get_pyarray_object_dim(in1) - 1) + " vs " + std::to_string(get_pyarray_object_dim(in2) - 1));
//     }

//     npy_intp *in1_shape = PyArray_DIMS(in1);
//     npy_intp *in2_shape = PyArray_DIMS(in2);
//     for(int i=0; i < dims; i++)
//     {
//         if(in1_shape[1+i] != in2_shape[1+i])
//         {
//             std::cout << "Tensor1 ";
//             print_shape(in1);
//             std::cout << "Tensor2 ";
//             print_shape(in2);
//             throw std::runtime_error("shapes of input tensors do not match");
//         }
//     }
// }

// void check_binary_channels(PyArrayObject *in)
// {
//     npy_intp *in_shape = PyArray_DIMS(in);
//     if (in_shape[0] != 2)
//     {
//         throw std::runtime_error("numpymaxflow currently only supports binary probability.");
//     }
// }

// void check_input_maxflow(PyArrayObject *image, PyArrayObject *prob, const int &num_dims)
// {
//     // check channels==2 for prob
//     check_binary_channels(prob);

//     // check spatial shapes match
//     check_spatial_shape_match(image, prob, num_dims-1);
// }

// void check_input_maxflow_interactive(PyArrayObject *image, PyArrayObject *prob, PyArrayObject *seed, const int &num_dims)
// {
//     // check channels==2 for prob and seeds
//     check_binary_channels(prob);
//     check_binary_channels(seed);

//     // check spatial shapes match
//     check_spatial_shape_match(image, prob, num_dims-1);
//     check_spatial_shape_match(image, seed, num_dims-1);
// }