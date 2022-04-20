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

#include "numpymaxflow.h"

static PyObject *
maxflow_wrapper(PyObject *self, PyObject *args)
{
    PyObject *image = NULL, *prob = NULL;

    // prepare arrays to read input args
    PyArrayObject *image_ptr = NULL, *prob_ptr = NULL;
    float lambda, sigma;

    // parse arguments into arrays and floats
    if (!PyArg_ParseTuple(args, "OOff", &image, &prob, &lambda, &sigma))
        return NULL;

    // read arrays from input args
    // old api
    // image_ptr = (PyArrayObject*)PyArray_FROM_OTF(image, NPY_FLOAT32, NPY_IN_ARRAY);
    // prob_ptr = (PyArrayObject*)PyArray_FROM_OTF(prob, NPY_FLOAT32, NPY_IN_ARRAY);
    // new api
    image_ptr = (PyArrayObject *)PyArray_FROM_OTF(image, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    prob_ptr = (PyArrayObject *)PyArray_FROM_OTF(prob, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

    if (image_ptr == NULL || prob_ptr == NULL)
    {
        return NULL;
    }

    // get number of dimensions
    int dim_image = PyArray_NDIM(image_ptr);
    // int dim_prob = PyArray_NDIM(prob_ptr);

    // npy_intp array of length nd showing length in each dim
    // could be 2D or 3D tensors of shapes
    // 2D: C x H x W  (3 dims)
    // 3D: C x D x H x W (4 dims)
    npy_intp *shape_image = PyArray_DIMS(image_ptr);
    npy_intp *shape_prob = PyArray_DIMS(prob_ptr);

    // if(dim_image > 3){
    //     cout << "the input dimension can only be 2 or 3"<<endl;
    //     return NULL;
    // }
    // if(dim_prob != 3){
    //     cout << "dimension of probabilily map should be 3"<<endl;
    //     return NULL;
    // }
    // if(shape_image[0] != shape_prob[0] || shape_image[1] != shape_prob[1]){
    //     cout << "image and probability map have different sizes"<<endl;
    //     return NULL;
    // }
    // if(shape_prob[2] != 2){
    //     cout << "probabilily map should have two channels"<<endl;
    //     return NULL;
    // }

    PyArrayObject *label_ptr;
    if (dim_image == 3) // 2D case with channels
    {
        npy_intp outshape[2];
        outshape[0] = shape_image[1];
        outshape[1] = shape_image[2];
        label_ptr = (PyArrayObject *)PyArray_SimpleNew(2, outshape, NPY_FLOAT32);

        // old api
        // maxflow2d_cpu((const float *) image_ptr->data, (const float *) prob_ptr->data, (float *) label_ptr->data,
        //      shape_image[0], shape_image[1], shape_image[2], lambda, sigma);
        // new api
        maxflow2d_cpu((const float *)PyArray_DATA(image_ptr), (const float *)PyArray_DATA(prob_ptr), (float *)PyArray_DATA(label_ptr),
                      shape_image[0], shape_image[1], shape_image[2], lambda, sigma);
    }
    else if (dim_image == 4) // 3D case with channels
    {
        npy_intp outshape[3];
        outshape[0] = shape_image[1];
        outshape[1] = shape_image[2];
        outshape[2] = shape_image[3];
        label_ptr = (PyArrayObject *)PyArray_SimpleNew(3, outshape, NPY_FLOAT32);

        // old api
        // maxflow3d_cpu((const float *) image_ptr->data, (const float *) prob_ptr->data, (float *) label_ptr->data,
        // shape_image[0], shape_image[1], shape_image[2], shape_image[3], lambda, sigma);
        // new api
        maxflow3d_cpu((const float *)PyArray_DATA(image_ptr), (const float *)PyArray_DATA(prob_ptr), (float *)PyArray_DATA(label_ptr),
                      shape_image[0], shape_image[1], shape_image[2], shape_image[3], lambda, sigma);
    }
    else
    {
        std::cout << "Unrecognised length dim";
    }
    Py_DECREF(image_ptr);
    Py_DECREF(prob_ptr);

    Py_INCREF(label_ptr);

    return PyArray_Return(label_ptr);

    // // could be 2D or 3D tensors of shapes
    // // 2D: C x H x W  (3 dims)
    // // 3D: C x D x H x W (4 dims)
    // const int num_dims = prob.dim();
    // check_input_maxflow(image, prob, num_dims);

    // // 2D case: C x H x W
    // if (num_dims == 4)
    // {
    //     return maxflow2d_cpu(image, prob, lambda, sigma);
    // }
    // // 3D case: C x D x H x W
    // else if (num_dims == 5)
    // {
    //     return maxflow3d_cpu(image, prob, lambda, sigma);
    // }
    // else
    // {
    //     throw std::runtime_error(
    //         "Library only supports 2D or 3D spatial inputs, received " + std::to_string(num_dims - 2) + "D inputs");
    // }
}

// torch::Tensor maxflow_interactive(const torch::Tensor &image, torch::Tensor &prob, const torch::Tensor &seed, const float &lambda, const float &sigma)
// {
//     // check input dimensions
//     // could be 2D or 3D tensors of shapes
//     // 2D: 1 x C x H x W  (4 dims)
//     // 3D: 1 x C x D x H x W (5 dims)
//     const int num_dims = prob.dim();
//     check_input_maxflow_interactive(image, prob, seed, num_dims);

//     // add interactive points to prob using seed locations
//     add_interactive_seeds(prob, seed, num_dims);

//     // 2D case: 1 x C x H x W
//     if (num_dims == 4)
//     {
//         return maxflow2d_cpu(image, prob, lambda, sigma);
//     }
//     // 3D case: 1 x C x D x H x W
//     else if (num_dims == 5)
//     {
//         return maxflow3d_cpu(image, prob, lambda, sigma);
//     }
//     else
//     {
//         throw std::runtime_error(
//             "Library only supports 2D or 3D spatial inputs, received " + std::to_string(num_dims - 2) + "D inputs");
//     }
// }
