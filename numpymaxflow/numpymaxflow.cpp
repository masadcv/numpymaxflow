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
    int connectivity=0;

    // parse arguments into arrays and floats
    if (!PyArg_ParseTuple(args, "OOff|i", &image, &prob, &lambda, &sigma, &connectivity))
    {
        return NULL;
    }

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
    int image_dims = PyArray_NDIM(image_ptr);
    int prob_dims = PyArray_NDIM(prob_ptr);

    // could be 2D or 3D tensors of shapes
    // 2D: C x H x W  (3 dims)
    // 3D: C x D x H x W (4 dims)
    // check shapes
    // check_input_maxflow(image_ptr, prob_ptr, prob_dims);
    // npy_intp array of length nd showing length in each dim
    npy_intp *shape_image = PyArray_DIMS(image_ptr);
    npy_intp *shape_prob = PyArray_DIMS(prob_ptr);

    if (shape_prob[0] != 2)
    {
        throw std::runtime_error("numpymaxflow currently only supports binary probability.");
    }

    if (image_dims != prob_dims)
    {
        throw std::runtime_error("dimensions of input tensors do not match " + std::to_string(image_dims - 1) + " vs " + std::to_string(prob_dims - 1));
    }

    for (int i = 0; i < prob_dims - 1; i++)
    {
        if (shape_image[1 + i] != shape_prob[1 + i])
        {
            std::cout << "Tensor1 ";
            for (int id = 0; id < prob_dims; id++)
            {
                std::cout << shape_image[id];
            }
            std::cout << "Tensor2 ";
            for (int id = 0; id < prob_dims; id++)
            {
                std::cout << shape_prob[id];
            }
            throw std::runtime_error("shapes of input tensors do not match");
        }
    }

    PyArrayObject *label_ptr;
    if (prob_dims == 3) // 2D case with channels
    {
        npy_intp outshape[3];
        outshape[0] = 1;
        outshape[1] = shape_image[1];
        outshape[2] = shape_image[2];
        label_ptr = (PyArrayObject *)PyArray_SimpleNew(3, outshape, NPY_FLOAT32);

        // old api
        // maxflow2d_cpu((const float *) image_ptr->data, (const float *) prob_ptr->data, (float *) label_ptr->data,
        //      shape_image[0], shape_image[1], shape_image[2], lambda, sigma, connectivity);
        // new api
        maxflow2d_cpu((const float *)PyArray_DATA(image_ptr), (const float *)PyArray_DATA(prob_ptr), (float *)PyArray_DATA(label_ptr),
                      shape_image[0], shape_image[1], shape_image[2], lambda, sigma, connectivity);
    }
    else if (prob_dims == 4) // 3D case with channels
    {
        npy_intp outshape[4];
        outshape[0] = 1;
        outshape[1] = shape_image[1];
        outshape[2] = shape_image[2];
        outshape[3] = shape_image[3];
        label_ptr = (PyArrayObject *)PyArray_SimpleNew(4, outshape, NPY_FLOAT32);

        // old api
        // maxflow3d_cpu((const float *) image_ptr->data, (const float *) prob_ptr->data, (float *) label_ptr->data,
        // shape_image[0], shape_image[1], shape_image[2], shape_image[3], lambda, sigma, connectivity);
        // new api
        maxflow3d_cpu((const float *)PyArray_DATA(image_ptr), (const float *)PyArray_DATA(prob_ptr), (float *)PyArray_DATA(label_ptr),
                      shape_image[0], shape_image[1], shape_image[2], shape_image[3], lambda, sigma, connectivity);
    }
    else
    {
        throw std::runtime_error(
            "numpymaxflow only supports 2D or 3D spatial inputs, received " + std::to_string(prob_dims - 1) + "D inputs");
    }
    Py_DECREF(image_ptr);
    Py_DECREF(prob_ptr);

    Py_INCREF(label_ptr);

    return PyArray_Return(label_ptr);
}

static PyObject *
maxflow_interactive_wrapper(PyObject *self, PyObject *args)
{
    PyObject *image = NULL, *prob = NULL, *seed = NULL;

    // prepare arrays to read input args
    PyArrayObject *image_ptr = NULL, *prob_ptr = NULL, *seed_ptr = NULL;
    float lambda, sigma;
    int connectivity;
    connectivity=0;

    // parse arguments into arrays and floats
    if (!PyArg_ParseTuple(args, "OOOff|i", &image, &prob, &seed, &lambda, &sigma, &connectivity))
    {
        return NULL;
    }

    // read arrays from input args
    // old api
    // image_ptr = (PyArrayObject*)PyArray_FROM_OTF(image, NPY_FLOAT32, NPY_IN_ARRAY);
    // prob_ptr = (PyArrayObject*)PyArray_FROM_OTF(prob, NPY_FLOAT32, NPY_IN_ARRAY);
    // new api
    image_ptr = (PyArrayObject *)PyArray_FROM_OTF(image, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    prob_ptr = (PyArrayObject *)PyArray_FROM_OTF(prob, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    seed_ptr = (PyArrayObject *)PyArray_FROM_OTF(seed, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

    if (image_ptr == NULL || prob_ptr == NULL || seed_ptr == NULL)
    {
        return NULL;
    }

    // get number of dimensions
    int image_dims = PyArray_NDIM(image_ptr);
    int prob_dims = PyArray_NDIM(prob_ptr);
    int seed_dims = PyArray_NDIM(prob_ptr);

    // could be 2D or 3D tensors of shapes
    // 2D: C x H x W  (3 dims)
    // 3D: C x D x H x W (4 dims)
    // check shapes
    // check_input_maxflow(image_ptr, prob_ptr, prob_dims);
    // npy_intp array of length nd showing length in each dim
    npy_intp *shape_image = PyArray_DIMS(image_ptr);
    npy_intp *shape_prob = PyArray_DIMS(prob_ptr);
    npy_intp *shape_seed = PyArray_DIMS(seed_ptr);

    if (shape_prob[0] != 2)
    {
        throw std::runtime_error("numpymaxflow currently only supports binary probability.");
    }

    if (shape_seed[0] != 2)
    {
        throw std::runtime_error("numpymaxflow currently only supports binary seeds.");
    }

    if (image_dims != prob_dims)
    {
        throw std::runtime_error("dimensions of input tensors do not match " + std::to_string(image_dims - 1) + " vs " + std::to_string(prob_dims - 1));
    }

    if (image_dims != seed_dims)
    {
        throw std::runtime_error("dimensions of input tensors do not match " + std::to_string(image_dims - 1) + " vs " + std::to_string(seed_dims - 1));
    }

    for (int i = 0; i < prob_dims - 1; i++)
    {
        if (shape_image[1 + i] != shape_prob[1 + i] || shape_image[1 + i] != shape_seed[1 + i])
        {
            std::cout << "Tensor1 ";
            for (int id = 0; id < prob_dims; id++)
            {
                std::cout << shape_image[id];
            }
            std::cout << "Tensor2 ";
            for (int id = 0; id < prob_dims; id++)
            {
                std::cout << shape_prob[id];
            }
            std::cout << "Tensor3 ";
            for (int id = 0; id < prob_dims; id++)
            {
                std::cout << shape_seed[id];
            }
            throw std::runtime_error("shapes of input tensors do not match");
        }
    }

    PyArrayObject *label_ptr;
    if (prob_dims == 3) // 2D case with channels
    {
        npy_intp outshape[2];
        outshape[0] = shape_image[1];
        outshape[1] = shape_image[2];
        label_ptr = (PyArrayObject *)PyArray_SimpleNew(2, outshape, NPY_FLOAT32);

        // old api
        // add_interactive_seeds_2d((float *) prob_ptr->data, (const float *) seed_ptr->data,
        //      shape_image[0], shape_image[1], shape_image[2]);
        // maxflow2d_cpu((const float *) image_ptr->data, (const float *) prob_ptr->data, (float *) label_ptr->data,
        //      shape_image[0], shape_image[1], shape_image[2], lambda, sigma);
        // new api
        add_interactive_seeds_2d((float *)PyArray_DATA(prob_ptr), (const float *)PyArray_DATA(seed_ptr),
                                 shape_image[0], shape_image[1], shape_image[2]);
        maxflow2d_cpu((const float *)PyArray_DATA(image_ptr), (const float *)PyArray_DATA(prob_ptr), (float *)PyArray_DATA(label_ptr),
                      shape_image[0], shape_image[1], shape_image[2], lambda, sigma, connectivity);
    }
    else if (prob_dims == 4) // 3D case with channels
    {
        npy_intp outshape[3];
        outshape[0] = shape_image[1];
        outshape[1] = shape_image[2];
        outshape[2] = shape_image[3];
        label_ptr = (PyArrayObject *)PyArray_SimpleNew(3, outshape, NPY_FLOAT32);

        // old api
        // add_interactive_seeds_3d((float *) prob_ptr->data, (const float *) seed_ptr->data,
        //      shape_image[0], shape_image[1], shape_image[2], shape_image[3]);
        // maxflow3d_cpu((const float *) image_ptr->data, (const float *) prob_ptr->data, (float *) label_ptr->data,
        //      shape_image[0], shape_image[1], shape_image[2], shape_image[3], lambda, sigma);
        // new api
        add_interactive_seeds_3d((float *)PyArray_DATA(prob_ptr), (const float *)PyArray_DATA(seed_ptr),
                                 shape_image[0], shape_image[1], shape_image[2], shape_image[3]);
        maxflow3d_cpu((const float *)PyArray_DATA(image_ptr), (const float *)PyArray_DATA(prob_ptr), (float *)PyArray_DATA(label_ptr),
                      shape_image[0], shape_image[1], shape_image[2], shape_image[3], lambda, sigma, connectivity);
    }
    else
    {
        throw std::runtime_error(
            "numpymaxflow only supports 2D or 3D spatial inputs, received " + std::to_string(prob_dims - 1) + "D inputs");
    }
    Py_DECREF(image_ptr);
    Py_DECREF(prob_ptr);
    Py_DECREF(seed_ptr);

    Py_INCREF(label_ptr);

    return PyArray_Return(label_ptr);
}
