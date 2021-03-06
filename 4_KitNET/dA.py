# Copyright (c) 2017 Yusuke Sugomori
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Portions of this code have been adapted from Yusuke Sugomori's code on GitHub: https://github.com/yusugomori/DeepLearning

import sys
import numpy
from utils import *  # math functions
import json

class dA_params:
    def __init__(self,n_visible = 5, n_hidden = 3, lr=0.001, corruption_level=0.0, gracePeriod = 10000, hiddenRatio=None):
        self.n_visible = n_visible   # num of units in visible (input) layer
        self.n_hidden = n_hidden     # num of units in hidden layer
        self.lr = lr
        self.corruption_level = corruption_level
        self.gracePeriod = gracePeriod
        self.hiddenRatio = hiddenRatio

class dA:
    def __init__(self, params):
        self.params = params

        if self.params.hiddenRatio is not None:
            self.params.n_hidden = int(numpy.ceil(self.params.n_visible*self.params.hiddenRatio)) #隐藏层数量

        # for 0-1 normlaization
        self.norm_max = numpy.ones((self.params.n_visible,)) * -numpy.Inf # tiny value 
        self.norm_min = numpy.ones((self.params.n_visible,)) * numpy.Inf  # Inf big data
        self.n = 0

        self.rng =  numpy.random.mtrand.RandomState(1234) #fixed random state
        #self.rng = numpy.random.seed(1234)

        a = 1. / self.params.n_visible #初始化权重1/a
        self.W = numpy.array(self.rng.uniform(  # initialize W uniformly
            low=-a,
            high=a,
            size=(self.params.n_visible, self.params.n_hidden)))  #初始化权重，[n_visible,n_hidden]

        self.hbias = numpy.zeros(self.params.n_hidden)  # initialize h bias 0 隐藏层偏置值
        self.vbias = numpy.zeros(self.params.n_visible)  # initialize v bias 0 输入层偏置值
        self.W_prime = self.W.T  #为减少训练量，隐藏层到输出层的权重为前一层权重的倒置


    def get_corrupted_input(self, input, corruption_level):
        assert corruption_level < 1

        return self.rng.binomial(size=input.shape,
                                 n=1,
                                 p=1 - corruption_level) * input # 二项分布 0，1取值 * 原输入矩阵，随机取0

    # Encode 获取隐藏层的输出
    def get_hidden_values(self, input):
        return sigmoid(numpy.dot(input, self.W) + self.hbias)

    # Decode #获得输出层的输出
    def get_reconstructed_input(self, hidden):
        return sigmoid(numpy.dot(hidden, self.W_prime) + self.vbias)

    def train(self, x):
        self.n = self.n + 1
        # update norms  #获取各维度最大值、最小值
        self.norm_max[x > self.norm_max] = x[x > self.norm_max]
        self.norm_min[x < self.norm_min] = x[x < self.norm_min]

        # 0-1 normalize #防止max=min，分母为0
        x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)

        if self.params.corruption_level > 0.0:
            tilde_x = self.get_corrupted_input(x, self.params.corruption_level)
        else:
            tilde_x = x
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        L_h2 = x - z
        L_h1 = numpy.dot(L_h2, self.W) * y * (1 - y)  # sigmoid 函数求导 sigmoid(y)' = y*(1-y)

        L_vbias = L_h2
        L_hbias = L_h1
        L_W = numpy.outer(tilde_x.T, L_h1) + numpy.outer(L_h2.T, y)

        self.W += self.params.lr * L_W
        self.hbias += self.params.lr * numpy.mean(L_hbias, axis=0)
        self.vbias += self.params.lr * numpy.mean(L_vbias, axis=0)
        return numpy.sqrt(numpy.mean(L_h2**2)) #the RMSE reconstruction error during training


    def reconstruct(self, x):  # 编码 + 解码
        y = self.get_hidden_values(x)  
        z = self.get_reconstructed_input(y)
        return z

    def execute(self, x): #returns MSE of the reconstruction of x
        if self.n < self.params.gracePeriod:
            return 0.0
        else:
            # 0-1 normalize
            x = (x - self.norm_min) / (self.norm_max - self.norm_min + 0.0000000000000001)
            z = self.reconstruct(x)
            rmse = numpy.sqrt(((x - z) ** 2).mean()) #MSE
            return rmse


    def inGrace(self):
        return self.n < self.params.gracePeriod
