# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# # from mmcv.cnn import xavier_init
# # from mmcv import Config
# # import numpy as np
# #
# # from ...registry import NECKS
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph import Linear, Sequential
from paddle.fluid.dygraph.nn import Conv2D, InstanceNorm
import paddle.fluid.dygraph.nn as nn
from model_tpn.ops import *


class Identity(fluid.dygraph.Layer):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ConvModule(fluid.dygraph.Layer):
    def __init__(
            self,
            inplanes,
            planes,
            kernel_size,
            stride,
            padding,
            bias=False,
            groups=1,
    ):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv3D(inplanes, planes, kernel_size, stride, padding, bias_attr=bias, groups=groups)
        self.bn = nn.BatchNorm(planes)
        self.relu = Relu()

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class AuxHead(fluid.dygraph.Layer):
    def __init__(
            self,
            inplanes=-1,
            planes=400,
            loss_weight=0.5
    ):
        super(AuxHead, self).__init__()
        self.convs = \
            ConvModule(inplanes, inplanes * 2, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.loss_weight = loss_weight
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(inplanes * 2, planes)

    # def init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             nn.init.constant_(m.bias, 0)
    #         if isinstance(m, nn.Conv3d):
    #             xavier_init(m, distribution='uniform')
    #         if isinstance(m, nn.BatchNorm3d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.fill_(0)

    def forward(self, x, target=None):
        if target is None:
            return None
        loss = dict()
        x = self.convs(x)
        # print(fluid.layers.adaptive_pool3d(x, 1).shape)
        x = fluid.layers.squeeze(fluid.layers.adaptive_pool3d(x, 1), [2, 3, 4])

        x = self.dropout(x)
        x = self.fc(x)

        target = fluid.layers.unsqueeze(target, 1)

        # print(x.shape)
        #
        # print('xxxxxxxxxxxx')
        # print(target.shape)

        loss['loss_aux'] = self.loss_weight * fluid.layers.softmax_with_cross_entropy(x, target)
        return loss


class TemporalModulation(fluid.dygraph.Layer):
    def __init__(self,
                 inplanes,
                 planes,
                 downsample_scale=8,
                 ):
        super(TemporalModulation, self).__init__()

        self.conv = nn.Conv3D(inplanes, planes, (3, 1, 1), (1, 1, 1), (1, 0, 0), bias_attr=False, groups=32)
        self.pool = MaxPool3D((downsample_scale, 1, 1), (downsample_scale, 1, 1), (0, 0, 0), ceil_mode=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class Upsampling(fluid.dygraph.Layer):
    def __init__(self,
                 scale=(2, 1, 1),
                 ):
        super(Upsampling, self).__init__()
        self.scale = scale

    def forward(self, x):
        # print('Upsampling',x.shape)
        # print(self.scale)
        x = fluid.layers.resize_trilinear(x, scale=1)
        # print(x.shape)
        return x


class Downampling(fluid.dygraph.Layer):
    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=(3, 1, 1),
                 stride=(1, 1, 1),
                 padding=(1, 0, 0),
                 bias=False,
                 groups=1,
                 norm=False,
                 activation=False,
                 downsample_position='after',
                 downsample_scale=(1, 2, 2),
                 ):
        super(Downampling, self).__init__()

        self.conv = nn.Conv3D(inplanes, planes, kernel_size, stride, padding, bias_attr=bias, groups=groups)
        self.norm = nn.BatchNorm(planes) if norm else None
        self.relu = Relu() if activation else None
        assert (downsample_position in ['before', 'after'])
        self.downsample_position = downsample_position
        self.pool = MaxPool3D(downsample_scale, downsample_scale, (0, 0, 0), ceil_mode=True)

    def forward(self, x):
        if self.downsample_position == 'before':
            x = self.pool(x)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.relu is not None:
            x = self.relu(x)
        if self.downsample_position == 'after':
            x = self.pool(x)

        return x


class LevelFusion(fluid.dygraph.Layer):
    def __init__(self,
                 in_channels=[1024, 1024],
                 mid_channels=[1024, 1024],
                 out_channels=2048,
                 ds_scales=[(1, 1, 1), (1, 1, 1)],
                 ):
        super(LevelFusion, self).__init__()

        ops = []
        num_ins = len(in_channels)
        for i in range(num_ins):
            op = Downampling(in_channels[i], mid_channels[i], kernel_size=(1, 1, 1), stride=(1, 1, 1),
                             padding=(0, 0, 0), bias=False, groups=32, norm=True, activation=True,
                             downsample_position='before', downsample_scale=ds_scales[i])
            ops.append(op)
            self.ops = Sequential(*ops)

        in_dims = np.sum(mid_channels)
        self.fusion_conv = Sequential(
            nn.Conv3D(in_dims, out_channels, 1, 1, 0, bias_attr=False),
            nn.BatchNorm(out_channels),
            Relu()
        )

    def forward(self, inputs):
        out = [self.ops[i](feature) for i, feature in enumerate(inputs)]
        out = fluid.layers.concat(out, 1)
        out = self.fusion_conv(out)
        return out


class SpatialModulation(fluid.dygraph.Layer):
    def __init__(
            self,
            inplanes=[1024, 2048],
            planes=2048,
    ):
        super(SpatialModulation, self).__init__()
        op = []
        for i, dim in enumerate(inplanes):

            ds_factor = planes // dim
            ds_num = int(np.log2(ds_factor))
            # print('ds_num',ds_num)
            # if ds_num < 1:
            #     # None
            #     op.append(Identity())
            #
            # else:
            #     for dsi in range(ds_num):
            #         print('dsi',dsi)
            #         in_factor = 2 ** dsi
            #         out_factor = 2 ** (dsi + 1)
            #         op.append(ConvModule(dim * in_factor, dim * out_factor, kernel_size=(1, 3, 3), stride=(1, 2, 2),
            #                              padding=(0, 1, 1), bias=False))

            # sub_layers.append(op)
            for j in range(ds_num):
                op.append(ConvModule(1024, 2048, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                     padding=(0, 1, 1), bias=False))
                op = Sequential(*op)
        self.spatial_modulation = Sequential(op)

    def forward(self, inputs):
        out = []
        # print('inputs',len(inputs))
        # for i, feature in enumerate(inputs):
        #     if isinstance(self.spatial_modulation[i], fluid.core.Layer):
        #         out_ = inputs[i]
        #         #print(len(self.spatial_modulation[i]))
        #         for j in range(len(self.spatial_modulation)):
        #         #for III, op in enumerate(self.spatial_modulation[i]):
        #
        #             out_ = self.spatial_modulation[j](out_)
        #         out.append(out_)
        #     else:
        #         out.append(self.spatial_modulation[i](inputs[i]))

        out0 = self.spatial_modulation(inputs[0])
        out.append(out0)
        out.append(inputs[1])

        return out


class TPN(fluid.dygraph.Layer):

    def __init__(self,
                 in_channels=[256, 512, 1024, 2048],
                 out_channels=256,
                 spatial_modulation_config=None,
                 temporal_modulation_config=None,
                 upsampling_config=None,
                 downsampling_config=None,
                 level_fusion_config=None,
                 aux_head_config=None,
                 mode=None):
        super(TPN, self).__init__()
        assert isinstance(in_channels, list)
        assert isinstance(out_channels, int)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.mode = mode
        # spatial_modulation_config = Config(spatial_modulation_config) if isinstance(spatial_modulation_config,
        #                                                                             dict) else spatial_modulation_config
        # temporal_modulation_config = Config(temporal_modulation_config) if isinstance(temporal_modulation_config,
        #                                                                               dict) else temporal_modulation_config
        # upsampling_config = Config(upsampling_config) if isinstance(upsampling_config, dict) else upsampling_config
        # downsampling_config = Config(downsampling_config) if isinstance(downsampling_config,
        #                                                                 dict) else downsampling_config
        # aux_head_config = Config(aux_head_config) if isinstance(aux_head_config, dict) else aux_head_config
        # level_fusion_config = Config(level_fusion_config) if isinstance(level_fusion_config,
        #                                                                 dict) else level_fusion_config

        # self.temporal_modulation_ops = nn.ModuleList()
        # self.upsampling_ops = nn.ModuleList()
        # self.downsampling_ops = nn.ModuleList()

        temp_modulation_ops = []
        temp_upsampling_ops = []
        temp_downsampling_ops = []
        for i in range(0, self.num_ins, 1):
            inplanes = in_channels[-1]
            planes = out_channels

            if temporal_modulation_config is not None:
                # overwrite the temporal_modulation_config
                # print(temporal_modulation_config)

                temporal_modulation_config['param']['downsample_scale'] = temporal_modulation_config['scales'][i]
                temporal_modulation_config['param']['inplanes'] = inplanes
                temporal_modulation_config['param']['planes'] = planes
                temporal_modulation = TemporalModulation(**temporal_modulation_config['param'])
                temp_modulation_ops.append(temporal_modulation)
            self.temporal_modulation_ops = Sequential(*temp_modulation_ops)

            if i < self.num_ins - 1:
                if upsampling_config is not None:
                    # overwrite the upsampling_config
                    upsampling = Upsampling(**upsampling_config)
                    temp_upsampling_ops.append(upsampling)
                self.upsampling_ops = Sequential(*temp_upsampling_ops)
                if downsampling_config is not None:
                    # overwrite the downsampling_config
                    downsampling_config['param']['inplanes'] = planes
                    downsampling_config['param']['planes'] = planes
                    downsampling_config['param']['downsample_scale'] = downsampling_config['scales']
                    downsampling = Downampling(**downsampling_config['param'])
                    temp_downsampling_ops.append(downsampling)
                self.downsampling_ops = Sequential(*temp_downsampling_ops)

        self.level_fusion_op = LevelFusion()  # **level_fusion_config
        self.spatial_modulation = SpatialModulation()  # **spatial_modulation_config
        out_dims = level_fusion_config['out_channels']

        # Two pyramids
        self.level_fusion_op2 = LevelFusion(**level_fusion_config)

        self.pyramid_fusion_op = Sequential(
            nn.Conv3D(out_dims * 2, 2048, 1, 1, 0, bias_attr=False),
            nn.BatchNorm(2048),
            Relu()
        )

        # overwrite aux_head_config
        if aux_head_config is not None:
            aux_head_config['inplanes'] = self.in_channels[-2]
            self.aux_head = AuxHead(**aux_head_config)
        else:
            self.aux_head = None

        # default init_weights for conv(msra) and norm in ConvModule
        # def init_weights(self):
        #     for m in self.modules():
        #         if isinstance(m, nn.Conv3D):
        #             xavier_init(m, distribution='uniform')
        #         if isinstance(m, nn.BatchNorm):
        #             m.weight.data.fill_(1)
        #             m.bias.data.fill_(0)

        # if self.aux_head is not None:
        #     self.aux_head.init_weights()

    def forward(self, inputs, target=None):
        loss = None

        # Auxiliary loss
        if self.mode == 'train':  #(self.aux_head is not None)
            loss = self.aux_head(inputs[-2], target)

        # Spatial Modulation
        outs = self.spatial_modulation(inputs)

        # Temporal Modulation
        outs = [self.temporal_modulation_ops[i](outs[i]) for i in range(len(self.temporal_modulation_ops))]

        temporal_modulation_outs = outs

        # Build top-down flow - upsampling operation
        if self.upsampling_ops is not None:
            for i in range(self.num_ins - 1, 0, -1):
                # print('upsampling_ops',outs[i].shape)
                outs[i - 1] = outs[i - 1] + self.upsampling_ops[i - 1](outs[i])

        # Get top-down outs
        topdownouts = self.level_fusion_op2(outs)
        outs = temporal_modulation_outs

        # Build bottom-up flow - downsampling operation
        if self.downsampling_ops is not None:
            for i in range(0, self.num_ins - 1, 1):
                outs[i + 1] = outs[i + 1] + self.downsampling_ops[i](outs[i])

        # Get bottom-up outs
        outs = self.level_fusion_op(outs)

        # fuse two pyramid outs
        # print('pyramid_fusion_op', len(topdownouts), len(outs))
        outs = self.pyramid_fusion_op(fluid.layers.concat([topdownouts, outs], 1))

        return outs, loss


if __name__ == '__main__':
    with fluid.dygraph.guard():
        x1 = np.random.uniform(-1, 1, [1, 1024, 8, 16, 16]).astype('float32')
        x1 = fluid.dygraph.to_variable(x1)
        x2 = np.random.uniform(-1, 1, [1, 2048, 8, 8, 8]).astype('float32')
        x2 = fluid.dygraph.to_variable(x2)
        x = (x1, x2)
        # y = np.random.uniform(-1, 1, [ 1, 3, 32, 32]).astype('float32')
        # y = fluid.dygraph.to_variable(y)
        # print('y',y)
        in_channels = [1024, 2048]
        out_channels = 1024

        spatial_modulation_config = dict(inplanes=[1024, 2048], planes=2048, )

        temporal_modulation_config = dict(scales=(32, 32),
                                          param=dict(inplanes=-1, planes=-1, downsample_scale=-1, ))

        upsampling_config = dict(scale=(1, 1, 1), )
        downsampling_config = dict(scales=(1, 1, 1),
                                   param=dict(inplanes=-1, planes=-1, downsample_scale=-1, ))
        level_fusion_config = dict(
            in_channels=[1024, 1024],
            mid_channels=[1024, 1024],
            out_channels=2048,
            ds_scales=[(1, 1, 1), (1, 1, 1)],
        )
        aux_head_config = dict(inplanes=-1, planes=400, loss_weight=0.5)

        model = TPN(in_channels=in_channels, out_channels=out_channels,
                    spatial_modulation_config=spatial_modulation_config,
                    temporal_modulation_config=temporal_modulation_config,
                    upsampling_config=upsampling_config, downsampling_config=downsampling_config,
                    level_fusion_config=level_fusion_config,
                    aux_head_config=dict(inplanes=-1, planes=400, loss_weight=0.5
                                         ))
        print('len(model.parameters())', len(model.parameters()))
        y = model(x)
        print(len(y))
        print('TPN shape', y[0].shape)
        print('loss', y[1])
