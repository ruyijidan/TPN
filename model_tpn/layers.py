# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from ...registry import SPATIAL_TEMPORAL_MODULES

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear, Sequential
from paddle.fluid.dygraph.nn import Conv2D, InstanceNorm
import paddle.fluid.dygraph.nn as nn
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear, BatchNorm, InstanceNorm
from model_tpn.ops import *


class AvgFusion(fluid.dygraph.Layer):
    def __init__(self, fusion_type='concat'):
        super(AvgFusion, self).__init__()
        assert fusion_type in ['add', 'avg', 'concat', 'concatadd', 'concatavg']
        self.fusion_type = fusion_type

    def init_weights(self):
        pass

    def forward(self, input):
        assert (isinstance(input, tuple))
        after_avgpool = [fluid.layers.adaptive_pool3d(each, 1) for each in input]

        if self.fusion_type == 'add':
            out = fluid.layers.reduce_mean(fluid.layers.concat(after_avgpool, -1), -1, keep_dim=True)

        elif self.fusion_type == 'avg':
            out = fluid.layers.reduce_mean(fluid.layers.concat(after_avgpool, -1), -1, keep_dim=True)

        elif self.fusion_type == 'concat':
            out = fluid.layers.concat(after_avgpool, 1)

        elif self.fusion_type == 'concatadd':
            out_first = fluid.layers.concat(after_avgpool[:-1], 1)
            out = fluid.layers.sum(fluid.layers.concat([out_first, after_avgpool[-1]], -1), -1, keep_dim=True)
        elif self.fusion_type == 'concatavg':
            out_first = fluid.layers.concat(after_avgpool[:-1], 1)
            out = fluid.layers.reduce_mean(fluid.layers.concat([out_first, after_avgpool[-1]], -1), -1, keep_dim=True)
        else:
            raise ValueError

        return out


'''
class SimpleSpatialModule(fluid.dygraph.Layer):
    def __init__(self, spatial_type='avg', spatial_size=7):
        super(SimpleSpatialModule, self).__init__()

        assert spatial_type in ['avg']
        self.spatial_type = spatial_type

        self.spatial_size = spatial_size if not isinstance(spatial_size, int) else (spatial_size, spatial_size)

        if self.spatial_type == 'avg':
            self.op = Pool2D(pool_size=self.spatial_size, pool_type="avg", pool_stride=1,
                             pool_padding=0, global_pooling=False)

    def init_weights(self):
        pass

    def forward(self, input):
        return self.op(input)
'''


class SimpleSpatialTemporalModule(fluid.dygraph.Layer):
    def __init__(self, spatial_type='avg', spatial_size=8, temporal_size=1 , mode='test'):
        super(SimpleSpatialTemporalModule, self).__init__()

        assert spatial_type in ['avg']
        self.spatial_type = spatial_type
        if mode == 'train':
            spatial_size = 7
        #spatial_size配置文件是7，推理时为8
        self.spatial_size = spatial_size if not isinstance(spatial_size, int) else (spatial_size, spatial_size)
        self.temporal_size = temporal_size
        self.pool_size = (self.temporal_size,) + self.spatial_size

        # print(self.spatial_size)
        # print(self.temporal_size)
        # print('pool_size',self.pool_size)
        if self.spatial_type == 'avg':
            self.op = AvgPool3D(self.pool_size, stride=1, padding=0)

    def init_weights(self):
        pass

    def forward(self, input):
        return self.op(input)


class _SimpleConsensus(fluid.dygraph.Layer):
    """Simplest segmental consensus module"""

    def __init__(self,
                 consensus_type='avg',
                 dim=1):
        super(_SimpleConsensus, self).__init__()

        assert consensus_type in ['avg']
        self.consensus_type = consensus_type
        self.dim = dim
        self.shape = None

    def forward(self, x):
        #print('Simplest', x.shape)
        self.shape = x.shape
        if self.consensus_type == 'avg':
            output = fluid.layers.reduce_mean(x, dim=self.dim, keep_dim=True)

        else:
            output = None
        return output

    def backward(self, grad_output):
        if self.consensus_type == 'avg':
            grad_in = fluid.layers.expand(grad_output, self.shape / float(self.shape[self.dim]))
        else:
            grad_in = None
        return grad_in


class SimpleConsensus(fluid.dygraph.Layer):
    def __init__(self, consensus_type='avg', dim=1):
        super(SimpleConsensus, self).__init__()

        assert consensus_type in ['avg']
        self.consensus_type = consensus_type
        self.dim = dim

    def init_weights(self):
        pass

    def forward(self, input):
        return _SimpleConsensus(self.consensus_type, self.dim)(input)


def main():
    with fluid.dygraph.guard():
        x = fluid.dygraph.to_variable(np.random.uniform(-1, 1, [8, 512, 8, 56, 56]).astype('float32'))
        res2 = fluid.dygraph.to_variable(np.random.uniform(-1, 1, [8, 512, 8, 56, 56]).astype('float32'))
        res3 = fluid.dygraph.to_variable(np.random.uniform(-1, 1, [8, 512, 8, 28, 28]).astype('float32'))
        res4 = fluid.dygraph.to_variable(np.random.uniform(-1, 1, [8, 512, 8, 14, 14]).astype('float32'))
        res5 = fluid.dygraph.to_variable(np.random.uniform(-1, 1, [8, 512, 8, 7, 7]).astype('float32'))
        feature = tuple([res2, res3, res4, res5])
        model = AvgFusion(fusion_type='add')
        out = model(feature)
        print(out.shape)


if __name__ == '__main__':
    # main()
    with fluid.dygraph.guard():
        x = np.random.uniform(-1, 1, [8, 2048, 1, 8, 8]).astype('float32')
        x = fluid.dygraph.to_variable(x)
        model0 = SimpleSpatialTemporalModule()
        y0 = model0(x)
        print('SimpleSpatialTemporalModule', y0.shape)
        y0=fluid.layers.unsqueeze(y0,0)
        print('input', y0.shape)
        model1 = SimpleConsensus()
        y1=model1(y0)
        print('SimpleConsensus',y1.shape)
