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

#include "common.h"
#include "graphcut.h"

// float l1distance(const float &in1, const float &in2)
// {
//     return std::abs(in1 - in2);
// }

// float l1distance(const float *in1, const float *in2, int size)
// {
//     float ret_sum = 0.0;
//     for (int c_i = 0; c_i < size; c_i++)
//     {
//         ret_sum += abs(in1[c_i] - in2[c_i]);
//     }
//     return ret_sum;
// }

float l2distance(const float &in1, const float &in2)
{
    return std::abs(in1 - in2);
}

float l2distance(const float *in1, const float *in2, int size)
{
    float ret_sum = 0.0;
    for (int c_i = 0; c_i < size; c_i++)
    {
        ret_sum += (in1[c_i] - in2[c_i]) * (in1[c_i] - in2[c_i]);
    }
    return std::sqrt(ret_sum);
}

float l2distance(const std::vector<float> &in1, const std::vector<float> &in2)
{
    int size = in1.size();
    float ret_sum = 0.0;
    for (int c_i = 0; c_i < size; c_i++)
    {
        ret_sum += (in1[c_i] - in2[c_i]) * (in1[c_i] - in2[c_i]);
    }
    return std::sqrt(ret_sum);
}

void maxflow2d_cpu(const float *image_ptr, const float *prob_ptr, float *label_ptr, const int &channel, const int &height, const int &width, const float &lambda, const float &sigma)
{
    const int Xoff[2] = {-1, 0};
    const int Yoff[2] = {0, -1};

    // prepare graph
    // initialise with graph(num of nodes, num of edges)
    GCGraph<float> g(height * width, 2 * height * width);

    float pval, qval, l2dis, n_weight, s_weight, t_weight, prob_bg, prob_fg;
    int pIndex, qIndex, cIndex;
    std::vector<float> pval_v(channel);
    std::vector<float> qval_v(channel);

    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            pIndex = g.addVtx();

            // format: C X H X W
            // index access: c * H * W + h * W + w
            // c = 0 for first channel
            cIndex = h * width + w;
            // avoid log(0)
            prob_bg = std::max(prob_ptr[cIndex], std::numeric_limits<float>::epsilon());

            // c = 1 for second channel
            cIndex = height * width + h * width + w;
            prob_fg = std::max(prob_ptr[cIndex], std::numeric_limits<float>::epsilon());
            s_weight = -log(prob_bg);
            t_weight = -log(prob_fg);

            g.addTermWeights(pIndex, s_weight, t_weight);
        }
    }

    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            pIndex = h * width + w;
            if (channel == 1)
            {
                pval = image_ptr[pIndex];
            }
            else
            {
                for (int c_i = 0; c_i < channel; c_i++)
                {
                    pval_v[c_i] = image_ptr[c_i * height * width + pIndex];
                }
            }

            for (int i = 0; i < 2; i++)
            {
                const int hn = h + Xoff[i];
                const int wn = w + Yoff[i];

                if (hn < 0 || wn < 0)
                {
                    continue;
                }

                qIndex = hn * width + wn;
                if (channel == 1)
                {
                    qval = image_ptr[qIndex];
                    l2dis = l2distance(pval, qval);
                }
                else
                {
                    for (int c_i = 0; c_i < channel; c_i++)
                    {
                        qval_v[c_i] = image_ptr[c_i * height * width + qIndex];
                    }
                    l2dis = l2distance(pval_v, qval_v);
                }
                n_weight = lambda * exp(-(l2dis * l2dis) / (2 * sigma * sigma));
                g.addEdges(qIndex, pIndex, n_weight, n_weight);
            }
        }
    }

    g.maxFlow();
    // float flow = g.maxFlow();
    // std::cout << "max flow: " << flow << std::endl;

    int idx = 0;
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            cIndex = h * width + w;
            label_ptr[cIndex] = g.inSourceSegment(idx) ? 1.0 : 0.0;
            idx++;
        }
    }
}

void maxflow3d_cpu(const float *image_ptr, const float *prob_ptr, float *label_ptr, const int &channel, const int &depth, const int &height, const int &width, const float &lambda, const float &sigma)
{
    const int Xoff[3] = {-1, 0, 0};
    const int Yoff[3] = {0, -1, 0};
    const int Zoff[3] = {0, 0, -1};

    // prepare graph
    // initialise with graph(num of nodes, num of edges)
    GCGraph<float> g(depth * height * width, 2 * depth * height * width);

    float pval, qval, l2dis, n_weight, s_weight, t_weight, prob_bg, prob_fg;
    int pIndex, qIndex, cIndex;
    std::vector<float> pval_v(channel);
    std::vector<float> qval_v(channel);
    for (int d = 0; d < depth; d++)
    {
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                pIndex = g.addVtx();

                // format: C X D X H X W
                // index access: c * D * H * W + d * H * W + h * W + w
                // c = 0 for first channel
                cIndex = d * height * width + h * width + w;

                // avoid log(0)
                prob_bg = std::max(prob_ptr[cIndex], std::numeric_limits<float>::epsilon());

                // c = 1 for second channel
                cIndex = depth * height * width + d * height * width + h * width + w;
                prob_fg = std::max(prob_ptr[cIndex], std::numeric_limits<float>::epsilon());
                s_weight = -log(prob_bg);
                t_weight = -log(prob_fg);

                g.addTermWeights(pIndex, s_weight, t_weight);
            }
        }
    }

    for (int d = 0; d < depth; d++)
    {
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                pIndex = d * height * width + h * width + w;
                if (channel == 1)
                {
                    pval = image_ptr[pIndex];
                }
                else
                {
                    for (int c_i = 0; c_i < channel; c_i++)
                    {
                        pval_v[c_i] = image_ptr[c_i * depth * height * width + pIndex];
                    }
                }

                for (int i = 0; i < 3; i++)
                {
                    const int dn = d + Xoff[i];
                    const int hn = h + Yoff[i];
                    const int wn = w + Zoff[i];

                    if (dn < 0 || hn < 0 || wn < 0)
                    {
                        continue;
                    }

                    qIndex = dn * height * width + hn * width + wn;
                    if (channel == 1)
                    {
                        qval = image_ptr[qIndex];
                        l2dis = l2distance(pval, qval);
                    }
                    else
                    {
                        for (int c_i = 0; c_i < channel; c_i++)
                        {
                            qval_v[c_i] = image_ptr[c_i * depth * height * width + qIndex];
                        }
                        l2dis = l2distance(pval_v, qval_v);
                    }
                    n_weight = lambda * exp(-(l2dis * l2dis) / (2 * sigma * sigma));
                    g.addEdges(qIndex, pIndex, n_weight, n_weight);
                }
            }
        }
    }

    g.maxFlow();

    // float flow = g.maxFlow();
    // std::cout << "max flow: " << flow << std::endl;

    int idx = 0;
    for (int d = 0; d < depth; d++)
    {
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                cIndex = d * height * width + h * width + w;
                label_ptr[cIndex] = g.inSourceSegment(idx) ? 1.0 : 0.0;
                idx++;
            }
        }
    }
}

void add_interactive_seeds_2d(float *prob_ptr, const float *seed_ptr, const int &channel, const int &height, const int &width)
{
    // implements Equation 7 from:
    //  Wang, Guotai, et al.
    //  "Interactive medical image segmentation using deep learning with image-specific fine tuning."
    //  IEEE TMI (2018).
    int cIndex;
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            cIndex = h * width + w;
            if (seed_ptr[cIndex] > 0)
            {
                prob_ptr[cIndex] = 1.0;
                prob_ptr[height * width + cIndex] = 0.0;
            }
            else if (seed_ptr[height * width + cIndex] > 0)
            {
                prob_ptr[cIndex] = 0.0;
                prob_ptr[height * width + cIndex] = 1.0;
            }
            else
            {
                continue;
            }
        }
    }
}

void add_interactive_seeds_3d(float *prob_ptr, const float *seed_ptr, const int &channel, const int &depth, const int &height, const int &width)
{
    int cIndex;
    for (int d = 0; d < depth; d++)
    {
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                cIndex = d * height * width + h * width + w;
                if (seed_ptr[cIndex] > 0)
                {
                    prob_ptr[cIndex] = 1.0;
                    prob_ptr[depth * height * width + cIndex] = 0.0;
                }
                else if (seed_ptr[depth * height * width + cIndex] > 0)
                {
                    prob_ptr[cIndex] = 0.0;
                    prob_ptr[depth * height * width + cIndex] = 1.0;
                }
                else
                {
                    continue;
                }
            }
        }
    }
}