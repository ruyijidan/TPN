# coding=utf-8

###### 欢迎使用脚本任务,让我们首选熟悉下一些使用规则吧 ###### 

# 数据集文件目录
datasets_prefix = '/root/paddlejob/workspace/train_data/datasets/'

# 数据集文件具体路径请在编辑项目状态下,通过左侧导航栏「数据集」中文件路径拷贝按钮获取
train_datasets =  datasets_prefix + 'data49371/k400.zip'

test_datasets =  datasets_prefix + 'data52214/val_256.zip'

# 输出文件目录. 任务完成后平台会自动把该目录所有文件压缩为tar.gz包，用户可以通过「下载输出」可以将输出信息下载到本地.
output_dir = "/root/paddlejob/workspace/output"

# 日志记录. 任务会自动记录环境初始化日志、任务执行日志、错误日志、执行脚本中所有标准输出和标准出错流(例如print()),用户可以在「提交」任务后,通过「查看日志」追踪日志信息.


import zipfile
import os.path
import os
import datetime

class ZFile(object):
    """
    文件压缩
    """

    def zip_file(self, fs_name, fz_name):
        """
        从压缩文件
        :param fs_name: 源文件名
        :param fz_name: 压缩后文件名
        :return:
        """
        flag = False
        if fs_name and fz_name:
            try:
                with zipfile.ZipFile(fz_name, mode='w', compression=zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(fs_name)
                    print(
                        "%s is running [%s] " %
                        (currentThread().getName(), fs_name))
                    print('压缩文件[{}]成功'.format(fs_name))
                if zipfile.is_zipfile(fz_name):
                    os.remove(fs_name)
                    print('删除文件[{}]成功'.format(fs_name))
                flag = True
            except Exception as e:
                print('压缩文件[{}]失败'.format(fs_name), str(e))

        else:
            print('文件名不能为空')
        return {'file_name': fs_name, 'flag': flag}

    def unzip_file(self, fz_name, path):
        """
        解压缩文件
        :param fz_name: zip文件
        :param path: 解压缩路径
        :return:
        """
        flag = False

        if zipfile.is_zipfile(fz_name):  # 检查是否为zip文件
            with zipfile.ZipFile(fz_name, 'r') as zipf:
                zipf.extractall(path)
                flag = True

        return {'file_name': fz_name, 'flag': flag}
        
'''
# 新建解压文件夹
dest_dir = "/root/paddlejob/workspace/datasets/"
os.system("mkdir " + dest_dir)

print('start to unzip k400')

# 解压数据集
if zipfile.is_zipfile(train_datasets):  # 检查是否为zip文件
    print('startING to unzip k400')
    with zipfile.ZipFile(train_datasets, 'r') as zipf:
        zipf.extractall(dest_dir)
        print('continue unzip...')
    print('unzip success.')
     
# start train decode
from argparse import ArgumentParser, REMAINDER
import paddle.distributed.launch as launch

from argparse import Namespace

# launch args
parser = ArgumentParser()
parser.add_argument("--cluster_node_ips", type=str, default="127.0.0.1",
    help='# Paddle cluster nodes ips, such as 192.168.0.16,192.168.0.17..')
parser.add_argument("--node_ip", type=str, default="127.0.0.1", help='The current node ip.')
parser.add_argument("--use_paddlecloud", action='store_true', default=False,
    help='wheter to use paddlecloud platform to run your multi-process job. If false, no need to set this argument.')
parser.add_argument("--started_port", type=int, default=None, help="The trainer's started port on a single node")
parser.add_argument("--print_config", type=bool, default=True, help='Print the config or not')
parser.add_argument("--selected_gpus", type=str, default=None,
    help="It's for gpu training and the training process will run on the selected_gpus, each "
         " process is bound to a single GPU. And if it's not set, this module will use all the gpu cards for training.")
parser.add_argument("--log_level", type=int, default=20, help='Logging level, default is logging.INFO')
parser.add_argument("--log_dir", type=str, default=None,
    help="The path for each process's log.If it's not set, the log will printed to default pipe.")

args = parser.parse_args()
args.training_script="train.py"
# config for single card
args.training_script_args=["--dataset_base_path", datasets_prefix + "data65/", "--output_base_path", output_dir + "model"]
# config for multi card，add --distributed
# args.training_script_args=["--distributed", "--dataset_base_path", datasets_prefix + "data65/", "--output_base_path", output_dir + "model"]
print(args)

launch.launch(args)

'''

import os

if __name__ == "__main__":
    
    print(os.getcwd())

    print('start')
    
    #os.system("pip install scipy==1.4.1")
    os.system("cd TPN && python -m paddle.distributed.launch --selected_gpus=0,1,2,3,4,5,6,7 train.py")
    #os.system("cd TPN && python eval.py")
