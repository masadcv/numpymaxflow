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
#include <vector>
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

int getIndex(const int &h, const int &w, const int &height, const int &width)
{
    return h * width + w;
}

int getIndex(const int &d, const int &h, const int &w, const int &depth, const int &height, const int &width)
{
    return d * height * width + h * width + w;
}

int getIndex(const int &c, const int &d, const int &h, const int &w, const int &channel, const int &depth, const int &height, const int &width)
{
    return c * depth * height * width + d * height * width + h * width + w;
}


void maxflow2d_cpu(const float *image_ptr, const float *prob_ptr, float *label_ptr, const int &channel, const int &height, const int &width, const float &lambda, const float &sigma, const int &connectivity)
{
    std::vector<int> Xoff, Yoff;
    int offsetLen;

    if (connectivity == 0)
    {
        std::cout << "numpymaxflow: warning no connectivity provided, falling back to default 4 connectivity" << std::endl;
    }

    if ((connectivity == 4) || (connectivity == 0))
    {
        Xoff = {-1, 0};
        Yoff = {0, -1};
        offsetLen = 2;
        // std::cout << "connectivity: " << connectivity << std::endl;
    }
    else if (connectivity == 8)
    {
        Xoff = {-1, 0, -1};
        Yoff = {0, -1, -1};
        offsetLen = 3;
        // std::cout << "connectivity: " << connectivity << std::endl;
    }
    else
    {
        throw std::runtime_error(
            "numpymaxflow only supports 4 or 8 connectivity for 2D spatial inputs, received connectivity = " + std::to_string(connectivity) + ".");
    };
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
            cIndex = getIndex(0, h, w, channel, height, width);
            // avoid log(0)
            prob_bg = std::max(prob_ptr[cIndex], std::numeric_limits<float>::epsilon());

            // c = 1 for second channel
            cIndex = getIndex(1, h, w, channel, height, width);
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
            pIndex = getIndex(h, w, height, width);
            if (channel == 1)
            {
                pval = image_ptr[pIndex];
            }
            else
            {
                for (int c_i = 0; c_i < channel; c_i++)
                {
                    cIndex = getIndex(c_i, h, w, channel, height, width);
                    pval_v[c_i] = image_ptr[cIndex];
                }
            }

            for (int i = 0; i < offsetLen; i++)
            {
                const int hn = h + Xoff[i];
                const int wn = w + Yoff[i];

                if (hn < 0 || wn < 0)
                {
                    continue;
                }

                qIndex = getIndex(hn, wn, height, width);
                if (channel == 1)
                {
                    qval = image_ptr[qIndex];
                    l2dis = l2distance(pval, qval);
                }
                else
                {
                    for (int c_i = 0; c_i < channel; c_i++)
                    {
                        cIndex = getIndex(c_i, hn, wn, channel, height, width);
                        qval_v[c_i] = image_ptr[cIndex];
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
            cIndex = getIndex(h, w, height, width);
            label_ptr[cIndex] = g.inSourceSegment(idx) ? 1.0 : 0.0;
            idx++;
        }
    }
}

void maxflow3d_cpu(const float *image_ptr, const float *prob_ptr, float *label_ptr, const int &channel, const int &depth, const int &height, const int &width, const float &lambda, const float &sigma, const int &connectivity)
{
    std::vector<int> Xoff, Yoff, Zoff;
    int offsetLen;
    
    if(connectivity == 0)
    {
        // no connectivity provided, issue warning and use default connectivity
        std::cout << "numpymaxflow: warning no connectivity provided, falling back to default 6 connectivity" << std::endl;
    }

    if ((connectivity == 6) || (connectivity == 0)) {
        Xoff = {-1, 0, 0};
        Yoff = {0, -1, 0};
        Zoff = {0, 0, -1};
        offsetLen = 3;
    }
    else if (connectivity == 18) {
        Xoff = {-1, 0, 0, -1, -1, 0};
        Yoff = {0, -1, 0, -1,  0, -1};
        Zoff = {0, 0, -1,  0, -1, -1};
        offsetLen = 6;
        // std::cout << "connectivity: " << connectivity << std::endl;

    }
    else if (connectivity == 26) {
        Xoff = {-1, 0, 0, -1, -1, 0,  -1};
        Yoff = {0, -1, 0, -1,  0, -1, -1};
        Zoff = {0, 0, -1,  0, -1, -1, -1};
        offsetLen = 7;
        // std::cout << "connectivity: " << connectivity << std::endl;

    }
    else {
        throw std::runtime_error(
            "numpymaxflow only supports 6, 18 or 26 connectivity for 3D spatial inputs, received connectivity = " + std::to_string(connectivity) + ".");
    };
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
                cIndex = getIndex(0, d, h, w, channel, depth, height, width);
                // avoid log(0)
                prob_bg = std::max(prob_ptr[cIndex], std::numeric_limits<float>::epsilon());

                // c = 1 for second channel
                cIndex = getIndex(1, d, h, w, channel, depth, height, width);
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
                pIndex = getIndex(d, h, w, depth, height, width);
                if (channel == 1)
                {
                    pval = image_ptr[pIndex];
                }
                else
                {
                    for (int c_i = 0; c_i < channel; c_i++)
                    {
                        cIndex = getIndex(c_i, d, h, w, channel, depth, height, width);
                        pval_v[c_i] = image_ptr[cIndex];
                    }
                }

                for (int i = 0; i < offsetLen; i++)
                {
                    const int dn = d + Xoff[i];
                    const int hn = h + Yoff[i];
                    const int wn = w + Zoff[i];

                    if (dn < 0 || hn < 0 || wn < 0)
                    {
                        continue;
                    }

                    qIndex = getIndex(dn, hn, wn, depth, height, width);
                    if (channel == 1)
                    {
                        qval = image_ptr[qIndex];
                        l2dis = l2distance(pval, qval);
                    }
                    else
                    {
                        for (int c_i = 0; c_i < channel; c_i++)
                        {
                            cIndex = getIndex(c_i, dn, hn, wn, channel, depth, height, width);
                            qval_v[c_i] = image_ptr[cIndex];
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
                cIndex = getIndex(d, h, w, depth, height, width);
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
    int cIndex0, cIndex1;
    for (int h = 0; h < height; h++)
    {
        for (int w = 0; w < width; w++)
        {
            cIndex0 = getIndex(0, h, w, channel, height, width);
            cIndex1 = getIndex(1, h, w, channel, height, width);
            if (seed_ptr[cIndex0] > 0)
            {
                prob_ptr[cIndex0] = 1.0;
                prob_ptr[cIndex1] = 0.0;
            }
            else if (seed_ptr[cIndex1] > 0)
            {
                prob_ptr[cIndex0] = 0.0;
                prob_ptr[cIndex1] = 1.0;
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
    int cIndex0, cIndex1;
    for (int d = 0; d < depth; d++)
    {
        for (int h = 0; h < height; h++)
        {
            for (int w = 0; w < width; w++)
            {
                cIndex0 = getIndex(0, d, h, w, channel, depth, height, width);
                cIndex1 = getIndex(1, d, h, w, channel, depth, height, width);
                if (seed_ptr[cIndex0] > 0)
                {
                    prob_ptr[cIndex0] = 1.0;
                    prob_ptr[cIndex1] = 0.0;
                }
                else if (seed_ptr[cIndex1] > 0)
                {
                    prob_ptr[cIndex0] = 0.0;
                    prob_ptr[cIndex1] = 1.0;
                }
                else
                {
                    continue;
                }
            }
        }
    }
}