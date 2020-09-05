import numpy as np
from model_tpn.cls_head import *
from model_tpn.layers import *
from model_tpn.ops import *
from model_tpn.resnet_slow import *
from model_tpn.tpn import *
from scipy.special import softmax

import logging

logger = logging.getLogger(__name__)


class Tpn_Model(fluid.dygraph.Layer):
    def __init__(self,
                 backbone,
                 necks=1,
                 spatial_temporal_module=1,
                 segmental_consensus=1,
                 fcn_testing=False,
                 flip=False,
                 cls_head=1,
                 cfg=None,
                 mode='train'):
        super(Tpn_Model, self).__init__()

        self.cfg = cfg
        # self.test_cfg = test_cfg
        self.fcn_testing = fcn_testing
        self.flip = flip
        self.video_length = 8
        self.crop_size = 224
        self.is_training = True
        self.mode = mode

        self.backbone = ResNet_SlowFast(depth=50, out_indices=[2, 3], inflate_freq=(0, 0, 1, 1),
                                        conv1_kernel_t=1, conv1_stride_t=1, pool1_kernel_t=1,
                                        pool1_stride_t=1, with_cp=True, )

        in_channels = [1024, 2048]
        out_channels = 1024

        spatial_modulation_config = dict(inplanes=[1024, 2048], planes=2048, )

        temporal_modulation_config = dict(scales=(32, 32),
                                          param=dict(inplanes=-1, planes=-1, downsample_scale=-1, ))

        upsampling_config = dict(scale=(1, 1, 1), )
        downsampling_config = dict(scales=(1, 1, 1), param=dict(inplanes=-1,
                                                                planes=-1, downsample_scale=-1, ))
        level_fusion_config = dict(in_channels=[1024, 1024], mid_channels=[1024, 1024],
                                   out_channels=2048, ds_scales=[(1, 1, 1), (1, 1, 1)], )
        if self.mode == 'train':
            aux_head_config = dict(inplanes=-1, planes=400, loss_weight=0.5)
        else:
            aux_head_config = None

        if necks is not None:

            self.necks = TPN(in_channels=in_channels, out_channels=out_channels,
                             spatial_modulation_config=spatial_modulation_config,
                             temporal_modulation_config=temporal_modulation_config,
                             upsampling_config=upsampling_config, downsampling_config=downsampling_config,
                             level_fusion_config=level_fusion_config,
                             aux_head_config=aux_head_config, mode=self.mode)
        else:
            self.necks = None

        if spatial_temporal_module is not None:
            # print('necks', spatial_temporal_module)

            self.spatial_temporal_module = SimpleSpatialTemporalModule(mode=self.mode)  # train or test
        else:
            raise NotImplementedError

        if segmental_consensus is not None:
            self.segmental_consensus = SimpleConsensus()
        else:
            raise NotImplementedError

        if cls_head is not None:
            self.cls_head = ClsHead()
        else:
            raise NotImplementedError

        # self.init_weights()

    def extract_feat(self, img_group):
        # img_group = img_group.reshape([-1,3,1,256,256])
        x = self.backbone(img_group)
        return x

    def build_input(self, use_dataloader=True):
        input_shape = [
            None, 3, self.video_length, self.crop_size, self.crop_size
        ]
        label_shape = [None, 1]

        data = fluid.data(
            name='train_data' if self.is_training else 'test_data',
            shape=input_shape,
            dtype='float32')
        if self.mode != 'infer':
            label = fluid.data(
                name='train_label' if self.is_training else 'test_label',
                shape=label_shape,
                dtype='int64')
        else:
            label = None

        if use_dataloader:
            assert self.mode != 'infer', \
                'dataloader is not recommendated when infer, please set use_dataloader to be false.'
            self.dataloader = fluid.io.DataLoader.from_generator(
                feed_list=[data, label], capacity=4, iterable=True)

        self.feature_input = [data]
        self.label_input = label

    def with_spatial_temporal_module(self):
        return hasattr(self, 'spatial_temporal_module') and self.spatial_temporal_module is not None

    def with_segmental_consensus(self):
        return hasattr(self, 'segmental_consensus') and self.segmental_consensus is not None

    def with_cls_head(self):
        return hasattr(self, 'cls_head') and self.cls_head is not None

    def get_config_from_sec(self, sec, item, default=None):
        if sec.upper() not in self.cfg:
            return default
        return self.cfg[sec.upper()].get(item, default)

    def optimizer(self):
        base_lr = 0.0001
        lr_decay = 0.1
        step_sizes = [150000, 150000, 100000]
        lr_bounds, lr_values = get_learning_rate_decay_list(base_lr, lr_decay,
                                                            step_sizes)
        learning_rate = fluid.layers.piecewise_decay(
            boundaries=lr_bounds, values=lr_values)

        momentum = 0.9
        use_nesterov = True
        l2_weight_decay = 0.0001
        logger.info(
            'Build up optimizer, \ntype: {}, \nmomentum: {}, \nnesterov: {}, \
                                         \nregularization: L2 {}, \nlr_values: {}, lr_bounds: {}'
                .format('Momentum', momentum, use_nesterov, l2_weight_decay,
                        lr_values, lr_bounds))
        optimizer = fluid.optimizer.Momentum(
            parameter_list=self.parameters(),
            learning_rate=learning_rate,
            momentum=momentum,
            use_nesterov=use_nesterov,
            regularization=fluid.regularizer.L2Decay(l2_weight_decay))
        return optimizer

    def feeds(self):
        return self.feature_input if self.mode == 'infer' else \
                     self.feature_input + [self.label_input]



    def forward(self, data):

        # print(data[0].shape)
        # print(data[1].shape)
        # assert num_modalities == 1
        if isinstance(data, list):
            img_group, gt_label = data
        else:
            img_group = data

        bs = img_group.shape[0]
        # print('self.mode', self.mode)
        if self.mode == 'train':
            img_group = fluid.layers.reshape(img_group,
                                             [-1, 3] + img_group.shape[2:])  # [-1,8,3,224,224]  [-1,32,3,224,224]
        else:
            # img_group = fluid.layers.reshape(img_group,[-1, 3, 1, 256, 256])
            img_group = fluid.layers.reshape(img_group,
                                             [-1, 3] + img_group.shape[2:])
        num_seg = img_group.shape[0] // bs
        #print('img_group', img_group.shape)
        x = self.extract_feat(img_group)

        # print('backbone out shape')
        # print(len(x))
        # # print(x[1].numpy())
        # for i in range(len(x)):
        #     print(x[i].shape)
        if self.mode == 'train':
            x, aux_losses = self.necks(x, fluid.layers.squeeze(gt_label, axes=[1]))
        else:
            x, _ = self.necks(x)

        # print('TPN', x.shape)
        # print('loss', aux_losses['loss_aux'].numpy())
        if self.with_spatial_temporal_module:
            x = self.spatial_temporal_module(x)
        # print('SimpleSpatialTemporalModule', x.shape)
        if self.with_segmental_consensus:
            x = fluid.layers.reshape(x, [-1, num_seg] + x.shape[1:])  # fluid.layers.reshape(x, [-1, 8] + x.shape[1:])
            # print('in', x.shape)
            x = self.segmental_consensus(x)
            x = fluid.layers.squeeze(x, [1])
        # print('SimpleConsensus', x.shape)
        losses = dict()
        if self.with_cls_head:
            cls_score = self.cls_head(x)
            # print('cls_score.shape',cls_score.shape)
            # gt_label = fluid.layers.squeeze(gt_label, axes=[1])

            # torch_cls = np.load('/home/j/Desktop/TPN-master/save_data/cls.npy')
            # print (np.sum(cls_score.numpy()-torch_cls))
            # prob = softmax(cls_score.numpy().squeeze())
            # idx = np.argsort(-prob)
            # print('idx', idx)
            if self.mode != 'train':
                return cls_score
            loss_cls = self.cls_head.loss(cls_score, gt_label)
            losses.update(loss_cls)
        if self.necks is not None:
            if aux_losses is not None:
                losses.update(aux_losses)

        # print('losses',losses)
        return [losses, cls_score, gt_label]

    def forward_train(self, data):
        # print(data[0].shape)
        # print(data[1].shape)
        # assert num_modalities == 1

        img_group, gt_label = data

        bs = img_group.shape[0]

        if self.mode == 'train':
            img_group = fluid.layers.reshape(img_group,
                                             [-1, 3] + img_group.shape[2:])  # [-1,8,3,224,224]  [-1,32,3,224,224]
        else:
            img_group = fluid.layers.reshape(img_group, [-1, 3, 1, 256, 256])

        num_seg = img_group.shape[0] // bs

        x = self.extract_feat(img_group)

        # print('backbone out shape')
        # print(len(x))
        # print(x[1].numpy())
        # for i in range(len(x)):
        #     print(x[i].shape)
        if self.necks is not None:
            x, aux_losses = self.necks(x, fluid.layers.squeeze(gt_label, axes=[1]))
        # print('TPN', x.shape)
        # print('loss', aux_losses['loss_aux'].numpy())
        if self.with_spatial_temporal_module:
            x = self.spatial_temporal_module(x)
        # print('SimpleSpatialTemporalModule', x.shape)
        if self.with_segmental_consensus:
            x = fluid.layers.reshape(x, [-1, num_seg] + x.shape[1:])  # fluid.layers.reshape(x, [-1, 8] + x.shape[1:])
            # print('in', x.shape)
            x = self.segmental_consensus(x)
            x = fluid.layers.squeeze(x, [1])
        # print('SimpleConsensus', x.shape)
        losses = dict()
        if self.with_cls_head:
            cls_score = self.cls_head(x)
            # gt_label = fluid.layers.squeeze(gt_label, axes=[1])

            loss_cls = self.cls_head.loss(cls_score, gt_label)
            losses.update(loss_cls)
        if self.necks is not None:
            if aux_losses is not None:
                losses.update(aux_losses)

        # print('losses',losses)
        return [losses, cls_score, gt_label]

    def forward_test(self,
                     num_modalities,
                     img_meta,
                     **kwargs):
        assert num_modalities == 1
        img_group = kwargs['img_group_0']
        print('img_group0', img_group.shape)
        bs = img_group.shape[0]
        # img_group = img_group.reshape((-1,) + img_group.shape[2:])

        img_group = fluid.layers.reshape(img_group, (-1,) + img_group.shape[2:])
        num_seg = img_group.shape[0] // bs
        print('img_group1', img_group.shape)
        if self.flip:
            img_group = self.extract_feat(fluid.layers.flip(img_group, [-1]))
        print('img_group2', img_group.shape)
        x = self.extract_feat(img_group)
        if self.necks is not None:
            x, _ = self.necks(x)
        if self.fcn_testing:
            if self.with_cls_head:
                x = self.cls_head(x)
                x = fluid.layers.reduce_mean(x, [2, 3, 4])
                x = fluid.layers.reduce_mean(x, 0, keep_dim=True)
                prob1 = fluid.layers.softmax(x).numpy()

                return prob1

        if self.with_spatial_temporal_module:
            x = self.spatial_temporal_module(x)
        if self.with_segmental_consensus:
            x = fluid.layers.reshape(x, (-1, num_seg) + x.shape[1:])  # x.reshape((-1, num_seg) + x.shape[1:])
            x = self.segmental_consensus(x)
            x = fluid.layers.squeeze(x, [1])
        if self.with_cls_head:
            x = self.cls_head(x)

        return x.cpu().numpy()


def get_learning_rate_decay_list(base_learning_rate, lr_decay, step_lists):
    lr_bounds = []
    lr_values = [base_learning_rate * 1]
    cur_step = 0
    for i in range(len(step_lists)):
        cur_step += step_lists[i]
        lr_bounds.append(cur_step)
        decay_rate = lr_decay ** (i + 1)
        lr_values.append(base_learning_rate * decay_rate)

    return lr_bounds, lr_values


if __name__ == '__main__':
    with fluid.dygraph.guard():
        # x = np.random.uniform(-1, 1, [8, 3, 32, 224, 224]).astype('float32')
        x = np.load('/home/j/Desktop/TPN-master/save_data/x.npy')
        x = fluid.dygraph.to_variable(x)
        label = fluid.dygraph.to_variable(np.random.uniform(-1, 1, [8, 1]).astype('int64'))
        print(x.shape)
        x = [x, label]
        model = Tpn_Model(None)
        model.eval()
        print('len(model.state_dict())0', len(model.state_dict()))

        weight, _ = fluid.load_dygraph('../ckpt/k400_tpn_r50f32s2')

        model_weights = model.state_dict()
        model_weights.update({k: v for k, v in weight.items()
                              if k in model_weights})
        model.load_dict(model_weights)

        print('len weight', len(weight))
        # model.load_dict(weight)
        y = model(x)
        print('len(model.state_dict())1', len(model.state_dict()))

        print(y)
        print(y[0], y[1], y[2])
        prob = softmax(y[1].numpy())
        print('prob', prob.shape)
        idx = np.argsort(-prob)
        print('idx', idx)

        # print(len(model.state_dict()))
        # print(model.state_dict().keys())
        #
        # for name, parameters in model.named_parameters():
        #     print(name, ':', parameters.shape)

        # backbone = ResNet_SlowFast(depth=50, out_indices=[2, 3], inflate_freq=(0, 0, 1, 1),
        #                         conv1_kernel_t=1, conv1_stride_t=1, pool1_kernel_t=1,
        #                         pool1_stride_t=1, with_cp=True, )
        #
        #
        # in_channels = [1024, 2048]
        # out_channels = 1024
        #
        # spatial_modulation_config = dict(inplanes=[1024, 2048], planes=2048, )
        #
        # temporal_modulation_config = dict(scales=(32, 32),
        #                                   param=dict(inplanes=-1, planes=-1, downsample_scale=-1, ))
        #
        # upsampling_config = dict(scale=(1, 1, 1), )
        # downsampling_config = dict(scales=(1, 1, 1),
        #                            param=dict(inplanes=-1, planes=-1, downsample_scale=-1, ))
        # level_fusion_config = dict(
        #     in_channels=[1024, 1024],
        #     mid_channels=[1024, 1024],
        #     out_channels=2048,
        #     ds_scales=[(1, 1, 1), (1, 1, 1)],
        # )
        # aux_head_config = dict(inplanes=-1, planes=400, loss_weight=0.5)
        #
        # tpn = TPN(in_channels=in_channels, out_channels=out_channels,
        #             spatial_modulation_config=spatial_modulation_config,
        #             temporal_modulation_config=temporal_modulation_config,
        #             upsampling_config=upsampling_config, downsampling_config=downsampling_config,
        #             level_fusion_config=level_fusion_config)
        #
        #
        # sstm = SimpleSpatialTemporalModule()
        # sc = SimpleConsensus()
        # cls = ClsHead()
        # print(len(backbone.parameters()))
        # y = backbone(x)
        #
        # print('backbone out shape')
        # print(len(y))
        # for i in range(len(y)):
        #     print(y[i].shape)
        #
        # y=tpn(y)
        # print('TPN',y[0].shape)
        # print('loss',y[1])
        #
        # y=sstm(y[0])
        # print('SimpleSpatialTemporalModule',y.shape)
        # #y=fluid.layers.unsqueeze(y,0)
        # y = fluid.layers.reshape(y,[-1, 8] + y.shape[1:])  #num_seg = img_group.shape[0] // bs
        # print('in',y.shape)
        # y=sc(y)
        # print('SimpleConsensus',y.shape)
        # y = fluid.layers.squeeze(y, [1])
        # y=cls(y)
        # print('cls_score',y.shape)
