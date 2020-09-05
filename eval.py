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
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
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
        '--batch_size',
        type=int,
        default=None,
        help='test batch size. None to use config file setting.')
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
        '--save_dir',
        type=str,
        default=os.path.join('data', 'evaluate_results'),
        help='output dir path, default to use ./data/evaluate_results')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=1,
        help='mini-batch interval to log.')
    args = parser.parse_args()
    return args


def test(args):
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    # place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)
    with fluid.dygraph.guard(place):
        # parse config
        config = parse_config(args.config)
        test_config = merge_configs(config, 'test', vars(args))
        print_configs(test_config, "Test")
        use_dali = test_config['TEST'].get('use_dali', False)

        # build model
        test_model = Tpn_Model(None, cfg=test_config, mode='eval')
        test_model.build_input(use_dataloader=False)

        # test_dataloader = test_model.dataloader()

        #if args.weights:
            # assert os.path.exists(
            #     args.weights), "Given weight dir {} not exist.".format(args.weights)
        weight, _ = fluid.load_dygraph(args.weights)
        model_weights = test_model.state_dict()
        model_weights.update({k: v for k, v in weight.items()
                              if k in model_weights})
        test_model.load_dict(model_weights)
        print('load model success')

        # get reader and metrics
        test_reader = get_reader(args.model_name.upper(), 'test', test_config)
        test_metrics = get_metrics(args.model_name.upper(), 'test', test_config)

        test_model.eval()

        epoch_period = []
        for test_iter, data in enumerate(test_reader()):
            cur_time = time.time()
            video_id = [items[-1] for items in data]

            # print(len(data))
            # print(len(data[0]),len(data[1]),len(data[2]),len(data[3]))
            # print( data[0][0].shape)
            # print(data[0][1])

            input_data=[]
            for i in range(len(data)):
                input_data.append(fluid.dygraph.to_variable(data[i][0]))

            # print(len(input_data))
            # print(input_data[0].shape)

            data_feed_in = fluid.layers.stack(input_data,0)
            #print('data_feed_in.shape',data_feed_in.shape)

            #input = fluid.dygraph.to_variable(data)

            test_outs = test_model(data_feed_in)

            # print(test_outs.shape)
            # print('video_id',np.stack(video_id, axis=0))

            test_outs = [test_outs.numpy(), np.stack(video_id, axis=0)]

            period = time.time() - cur_time
            epoch_period.append(period)
            test_metrics.accumulate(test_outs)

            # metric here
            if args.log_interval > 0 and test_iter % args.log_interval == 0:
                info_str = '[EVAL] Batch {}'.format(test_iter)
                test_metrics.calculate_and_log_out(test_outs, info_str)

        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)
        test_metrics.finalize_and_log_out("[EVAL] eval finished. ", args.save_dir)


if __name__ == "__main__":
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    check_cuda(args.use_gpu)
    check_version()
    logger.info(args)

    test(args)
