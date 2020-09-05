import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear, Sequential
from paddle.fluid.dygraph.nn import Conv2D, InstanceNorm
import paddle.fluid.dygraph.nn as nn
from paddle.fluid.param_attr import ParamAttr


class Relu(fluid.dygraph.Layer):
    def __init__(self):
        super(Relu, self).__init__()

    def forward(self, x):
        return fluid.layers.relu(x)


class MaxPool3D(fluid.dygraph.Layer):
    def __init__(self, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(2, 1, 1),
                 pool_type='max', ceil_mode=False):
        super(MaxPool3D, self).__init__()

        self.pool_size = kernel_size
        self.pool_stride = stride
        self.pool_padding = padding
        self.pool_type = pool_type
        self.ceil_mode = ceil_mode

    def forward(self, x):
        return fluid.layers.pool3d(x, pool_size=self.pool_size, pool_stride=self.pool_stride
                                   , pool_padding=self.pool_padding, pool_type=self.pool_type)


class AvgPool3D(fluid.dygraph.Layer):
    def __init__(self, kernel_size, stride, padding, pool_type='avg'):
        super(AvgPool3D, self).__init__()

        self.pool_size = kernel_size
        self.pool_stride = stride
        self.pool_padding = padding
        self.pool_type = pool_type

    def forward(self, x):
        return fluid.layers.pool3d(x, pool_size=self.pool_size, pool_stride=self.pool_stride
                                   , pool_padding=self.pool_padding, pool_type=self.pool_type)


class Batch_Norm3D(fluid.dygraph.Layer):
    def __init__(self, is_test=False,momentum=0.1,in_place=True):
        super(Batch_Norm3D, self).__init__()
        self.name='bn1'
        self.bn1=ParamAttr(
            name=self.name + "_scale",
            regularizer=fluid.regularizer.L2Decay(
            ))
    def forward(self, x):

        return fluid.layers.batch_norm(x, act=None, is_test=False, momentum=0.9, epsilon=1e-05,
                                       param_attr=None, bias_attr=None, data_layout='NCHW',
                                       in_place=False, name=None, moving_mean_name=None,
                                       moving_variance_name=None, do_model_average_for_mean_and_var=False,
                                       use_global_stats=False)
