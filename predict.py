#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import time
import logging
import argparse
import ast
import numpy as np
from scipy.special import softmax

try:
    import cPickle as pickle
except:
    import pickle
import paddle.fluid as fluid

from utils.config_utils import *
from model_tpn.model import Tpn_Model
from reader import get_reader
from metrics import get_metrics
from utils.utility import check_cuda
from utils.utility import check_version
from paddle.fluid import profiler
from collections import OrderedDict

logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        type=str,
        default='NONLOCAL',
        help='name of model to train.')
    parser.add_argument(
        '--config',
        type=str,
        default='./configs/tpn.yaml',
        help='path to config file of model')
    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--weights',
        type=str,
        default='./ckpt/k400_tpn_r50f32s2',
        help='weight path, None to automatically download weights provided by Paddle.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1,
        help='sample number in a batch for inference.')
    parser.add_argument(
        '--filelist',
        type=str,
        default=None,
        help='path to inferenece data file lists file.')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval to log.')
    parser.add_argument(
        '--infer_topk',
        type=int,
        default=20,
        help='topk predictions to restore.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default=os.path.join('data', 'predict_results'),
        help='directory to store results')
    parser.add_argument(
        '--video_path',
        type=str,
        default=None,
        help='directory to store results')
    args = parser.parse_args()
    return args


def infer(args):
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        # parse config
        config = parse_config(args.config)
        infer_config = merge_configs(config, 'infer', vars(args))
        print_configs(infer_config, "Infer")

        infer_model = Tpn_Model(None, cfg=infer_config, mode='test')
        infer_model.build_input(use_dataloader=False)

        infer_model.eval()


        filelist = args.filelist or infer_config.INFER.filelist
        filepath = args.video_path or infer_config.INFER.get('filepath', '')
        if filepath != '':
            assert os.path.exists(filepath), "{} not exist.".format(filepath)
        else:
            assert os.path.exists(filelist), "{} not exist.".format(filelist)

        # get infer reader
        infer_reader = get_reader(args.model_name.upper(), 'infer', infer_config)

        # if args.weights:
        # assert os.path.exists(
        #     args.weights), "Given weight dir {} not exist.".format(args.weights)
        weight, _ = fluid.load_dygraph(args.weights)
        model_weights = infer_model.state_dict()
        model_weights.update({k: v for k, v in weight.items()
                              if k in model_weights})
        infer_model.load_dict(model_weights)
        print('load model success')



        infer_metrics = get_metrics(args.model_name.upper(), 'infer', infer_config)
        infer_metrics.reset()

        periods = []
        cur_time = time.time()

        for infer_iter, data in enumerate(infer_reader()):
            # print(infer_iter, data)
            data_feed_in = [items[:-1] for items in data]
            video_id = [items[-1] for items in data]
            input = fluid.dygraph.to_variable(data_feed_in[0][0])
            #print(input.numpy().shape)
            input = fluid.layers.unsqueeze(input, 0)

            # x = np.load('/home/j/Desktop/TPN-master/save_data/x.npy')
            # x = fluid.dygraph.to_variable(x)

            data_feed_in = input

            # print('input shape', data_feed_in[0].shape)

            infer_outs = infer_model(data_feed_in)


            # print("infer_outs")
            # print(infer_outs)
            # print(infer_outs[1].numpy().shape)
            # infer_outs = [infer_outs[1].numpy(),video_id]

            pred = softmax(infer_outs.numpy())

            # #分类结果
            # prob = softmax(infer_outs.numpy().squeeze())
            #
            # idx = np.argsort(-prob)
            # #print('idx', idx)
            # for i in range(0, 5):
            #     print('{:.3f} -> {}'.format(prob[idx[i]], [idx[i]]))


            infer_result_list = [pred, video_id]  # [item[1].numpy() for item in infer_outs] + [video_id]

            prev_time = cur_time
            cur_time = time.time()
            period = cur_time - prev_time
            periods.append(period)

            infer_metrics.accumulate(infer_result_list)

            if args.log_interval > 0 and infer_iter % args.log_interval == 0:
                logger.info('Processed {} samples'.format((infer_iter + 1) * len(
                    video_id)))

        logger.info('[INFER] infer finished. average time: {}'.format(
            np.mean(periods)))

        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)

        infer_metrics.finalize_and_log_out(savedir=args.save_dir)


if __name__ == "__main__":
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    check_cuda(args.use_gpu)
    check_version()
    logger.info(args)

    infer(args)
