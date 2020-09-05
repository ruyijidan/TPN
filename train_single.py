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
import argparse
import ast
import logging
import numpy as np
import paddle.fluid as fluid

from utils.train_utils import *
from model_tpn.model import Tpn_Model
from utils.config_utils import *
from reader import get_reader
from metrics import get_metrics
from utils.utility import check_cuda
from utils.utility import check_version
from paddle.fluid import profiler
from collections import OrderedDict
from scipy.special import softmax


logging.root.handlers = []
FORMAT = '[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser("Paddle Video train script")
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
        help='training batch size. None to use config file setting.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='learning rate use for training. None to use config file setting.')
    parser.add_argument(
        '--pretrain',
        type=str,
        default=None,
        help='path to pretrain weights. None to use default weights path in  ~/.paddle/weights.'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='path to resume training based on previous checkpoints. '
             'None for not resuming any checkpoints.')

    parser.add_argument(
        '--selected_gpus',
        type=str,
        default='0,1,2,3',
        help='multi gpus number.')

    parser.add_argument(
        '--use_gpu',
        type=ast.literal_eval,
        default=True,
        help='default use gpu.')
    parser.add_argument(
        '--no_memory_optimize',
        action='store_true',
        default=False,
        help='whether to use memory optimize in train')
    parser.add_argument(
        '--epoch',
        type=int,
        default=10,
        help='epoch number, 0 for read from config file')
    parser.add_argument(
        '--valid_interval',
        type=int,
        default=1,
        help='validation epoch interval, 0 for no validation.')
    parser.add_argument(
        '--save_dir',
        type=str,
        default=os.path.join('data', 'checkpoints'),  # ./data/checkpoints
        help='directory name to save train snapshoot')
    parser.add_argument(
        '--log_interval',
        type=int,
        default=10,
        help='mini-batch interval to log.')
    parser.add_argument(
        '--fix_random_seed',
        type=ast.literal_eval,
        default=False,
        help='If set True, enable continuous evaluation job.')
    # NOTE: args for profiler, used for benchmark
    parser.add_argument(
        '--profiler_path',
        type=str,
        default='./',
        help='the path to store profiler output file. used for benchmark.')
    parser.add_argument(
        '--is_profiler',
        type=int,
        default=0,
        help='the switch profiler. used for benchmark.')
    args = parser.parse_args()
    return args


def train(args):
    #获取GPU
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)
    print(place)

    with fluid.dygraph.guard(place):
        #多卡上下文
        strategy = fluid.dygraph.parallel.prepare_context()
        print('strategy',strategy)

        # parse config
        config = parse_config(args.config)
        train_config = merge_configs(config, 'train', vars(args))
        valid_config = merge_configs(config, 'valid', vars(args))
        print_configs(train_config, 'Train')
        print(train_config)

        # if args.fix_random_seed:
        #     startup.random_seed = 1000
        #     train_prog.random_seed = 1000

        train_model = Tpn_Model(None, cfg=train_config,mode='train')  # models.get_model(args.model_name, train_config, mode='train')


        valid_model = Tpn_Model(None)  # models.get_model(args.model_name, valid_config, mode='valid')
        train_model.build_input()
        train_dataloader = train_model.dataloader()
        opt = train_model.optimizer()

        # load weights
        weight, _ = fluid.load_dygraph('./ckpt/k400_tpn_r50f32s2')
        model_weights = train_model.state_dict()
        model_weights.update({k: v for k, v in weight.items()
                              if k in model_weights})
        train_model.load_dict(model_weights)
        print('load model success')

        # 模型并行
        #train_model = fluid.dygraph.parallel.DataParallel(train_model, strategy)

        log_interval = args.log_interval
        is_profiler = args.is_profiler
        profiler_path = args.profiler_path
        trainer_id = 0
        fix_random_seed = args.fix_random_seed
        save_dir = args.save_dir
        save_model_name = args.model_name



        # if args.resume:
        #     # if resume weights is given, load resume weights directly
        #     assert os.path.exists(args.resume + '.pdparams'), \
        #         "Given resume weight dir {}.pdparams not exist.".format(args.resume)
        #     fluid.load(train_prog, model_path=args.resume, executor=exe)
        # else:
        #     # if not in resume mode, load pretrain weights
        #     if args.pretrain:
        #         assert os.path.exists(args.pretrain), \
        #             "Given pretrain weight dir {} not exist.".format(args.pretrain)
        #     pretrain = args.pretrain or train_model.get_pretrain_weights()
        #     if pretrain:
        #         train_model.load_pretrain_params(exe, pretrain, train_prog, place)

        # get reader
        bs_denominator = 1
        if args.use_gpu:
            # check number of GPUs
            gpus = os.getenv("CUDA_VISIBLE_DEVICES", "")
            if gpus == "":
                pass
            else:
                gpus = gpus.split(",")
                num_gpus = len(gpus)
                assert num_gpus == train_config.TRAIN.num_gpus, \
                    "num_gpus({}) set by CUDA_VISIBLE_DEVICES " \
                    "shoud be the same as that " \
                    "set in {}({})".format(
                        num_gpus, args.config, train_config.TRAIN.num_gpus)
            bs_denominator = train_config.TRAIN.num_gpus

        train_config.TRAIN.batch_size = int(train_config.TRAIN.batch_size /
                                            bs_denominator)
        valid_config.VALID.batch_size = int(valid_config.VALID.batch_size /
                                            bs_denominator)
        train_reader = get_reader(args.model_name.upper(), 'train', train_config)
        valid_reader = get_reader(args.model_name.upper(), 'valid', valid_config)

        # get metrics
        train_metrics = get_metrics(args.model_name.upper(), 'train', train_config)
        valid_metrics = get_metrics(args.model_name.upper(), 'valid', valid_config)

        epochs = args.epoch #or train_model.epoch_num()





        print()

        train_dataloader.set_sample_list_generator(train_reader, places=place)
        # valid_dataloader.set_sample_list_generator(valid_reader, places=exe_places)

        ##多GPU数据读取，必须确保每个进程读取的数据是不同的
        #train_dataloader = fluid.contrib.reader.distributed_batch_reader(train_dataloader)

        train_model.train()

        for epoch in range(epochs):
            log_lr_and_step()
            train_iter = 0
            epoch_periods = []
            cur_time = time.time()
            for data in train_dataloader():
                train_outs = train_model(data)
                losses, _, _ = train_outs
                log_vars = OrderedDict()
                for loss_name, loss_value in losses.items():
                    # print(loss_name, ':', loss_value.numpy())
                    log_vars[loss_name] = fluid.layers.reduce_mean(loss_value)
                    # print(loss_name, ':', log_vars[loss_name].numpy())

                loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
                # print('total loss', loss.numpy())

                train_outs = [loss.numpy(), train_outs[1].numpy(), train_outs[2].numpy()]

                # print(train_outs[0])
                # print(train_outs[1].shape)
                # print(train_outs[2])
                #
                # # # #分类结果
                # prob = softmax(train_outs[1].squeeze())
                #
                # idx = np.argsort(-prob)
                # print('idx', idx)
                # for i in range(0, 5):
                #     print('{:.3f} -> {}'.format(prob[idx[i]], [idx[i]]),train_outs[2])


                avg_loss = loss
                # 多GPU训练需要对Loss做出调整，并聚合不同设备上的参数梯度
                #avg_loss = train_model.scale_loss(avg_loss)

                avg_loss.backward()
                # 多GPU
                #train_model.apply_collective_grads()

                opt.minimize(avg_loss)
                train_model.clear_gradients()
                period = time.time() - cur_time
                epoch_periods.append(period)
                timeStamp = time.time()
                localTime = time.localtime(timeStamp)
                strTime = time.strftime("%Y-%m-%d %H:%M:%S", localTime)

                if log_interval > 0 and (train_iter % log_interval == 0):
                    train_metrics.calculate_and_log_out(train_outs, \
                                                        info='[TRAIN {}] Epoch {}, iter {}, time {}, '.format(strTime,
                                                                                                              epoch,
                                                                                                              train_iter,
                                                                                                              period))

                    # print('[TRAIN {}] Epoch {}, iter {}, time {}, total_loss {}, loss_cls {},loss_aux {}'.
                    #       format(strTime, epoch, train_iter, period, loss.numpy(),
                    #              log_vars['loss_cls'].numpy(), log_vars['loss_aux'].numpy()
                    #              ))
                train_iter += 1
                cur_time = time.time()

                # NOTE: profiler tools, used for benchmark
                if is_profiler and epoch == 0 and train_iter == log_interval:
                    profiler.start_profiler("All")
                elif is_profiler and epoch == 0 and train_iter == log_interval + 5:
                    profiler.stop_profiler("total", profiler_path)
                    return
            if len(epoch_periods) < 1:
                logger.info(
                    'No iteration was executed, please check the data reader')
                sys.exit(1)

            logger.info('[TRAIN] Epoch {} training finished, average time: {}'.
                        format(epoch, np.mean(epoch_periods[1:])))

            # if trainer_id == 0:
            #     save_model(exe, train_prog, save_dir, save_model_name,
            #                "_epoch{}".format(epoch))
            # if compiled_test_prog and valid_interval > 0 and (
            #         epoch + 1) % valid_interval == 0:
            #     test_with_dataloader(exe, compiled_test_prog, test_dataloader,
            #                          test_fetch_list, test_metrics, log_interval,
            #                          save_model_name)

        if trainer_id == 0:
            # save_model(exe, train_prog, save_dir, save_model_name)
            fluid.save_dygraph(train_model.state_dict(), "{}/{}".format(save_dir, save_model_name))
            fluid.save_dygraph(opt.state_dict(), "{}/{}}".format(save_dir, save_model_name))
        # when fix_random seed for debug
        if fix_random_seed:
            cards = os.environ.get('CUDA_VISIBLE_DEVICES')
            gpu_num = len(cards.split(","))
            print("kpis\ttrain_cost_card{}\t{}".format(gpu_num, loss))
            print("kpis\ttrain_speed_card{}\t{}".format(gpu_num,
                                                        np.mean(epoch_periods)))


if __name__ == "__main__":
    args = parse_args()
    # check whether the installed paddle is compiled with GPU
    check_cuda(args.use_gpu)
    check_version()
    logger.info(args)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)


    train(args)
