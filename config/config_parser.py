import os
import json
import torch

import os
import json
import torch


class DGCONFIG(object):
    r"""The basic DGCONFIG for generating a list of hyperparameters,.
    This class defines a basic template class for CONFIG.
    The following steps will are executed automatically:

      1. Check whether args is NONE. If true, goto 3.
      2. Call ``parse_external_config()`` Personalize custom arguments based on args and other_args.
      3. Call ``parse_config_file()`` If the default model parameters are used, the function will read the parameters according to the specified model.
      4. Call ``load_default_config()`` Load parameters according to the file address provided by config_file and goto 6.
      5. Call ``init_device()`` Initialize the device.
      6. Done.
          """
    def __init__(self, args,config_file=None,other_args=None,):
        self._args=vars(args)
        self._other_args=None
        self._config_file=config_file
        if other_args is not None:
            self._other_args=vars(other_args)
        self._config = {}
        self._parse_external_config()
        self._parse_config_file()
        self._load_default_config()
        self._init_device()

    def _parse_external_config(self):
        if self.args is not None :
            for key in self.args:
                if key not in self.config:
                    self.config[key] = self.args[key]

        if self.other_args is not None:
            # TODO: 这里可以设计加入参数检查，哪些参数是允许用户通过命令行修改的
            for key in self.other_args:
                if key not in self.config:
                    self.config[key] = self.other_args[key]

    def _parse_config_file(self):
        if config_file is  None:
            # TODO: 对 config file 的格式进行检查
            model=self.config.get("model",None)
            assert model, "The file pointed to by config_file was not found and the default model was not specified"

            if self.config.get("task",None) is None:
                raise ValueError('the parameter task should not be None!')

            if self.config.get("dataset",None) is None:
                raise ValueError('the parameter dataset should not be None!')

            Path = './config/config_data/Model_parameters/{}.json'.format(model)
            if os.path.exists(Path):
                with open(Path, 'r') as f:
                    x = json.load(f)
                    for key in x:
                        if key not in config:
                            config[key] = x[key]
            else:
                raise FileNotFoundError(
                    'Config file {}.json is not found. Please ensure \
                    the config file is in the root dir and is a JSON \
                    file.'.format(config_file))

    def _load_default_config(self):
        pass

    def _init_device(self):
        gpu_id = self.config.get('gpu', -1)
        self.config['device'] = torch.device(
            "cuda:%d" % gpu_id if gpu_id>-1  else "cpu")

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getitem__(self, key):
        if key in self.config:
            return self.config[key]
        else:
            raise KeyError('{} is not in the config'.format(key))

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, key):
        return key in self.config

    # 支持迭代操作
    def __iter__(self):
        return self.config.__iter__()

    @property
    def config(self):
        r"""return config.
        """
        return self._config

    @property
    def args(self):
        r"""return args.
               """
        return self._args

    @property
    def other_args(self):
        r"""return other_args.
               """
        return self._other_args

    @property
    def config_file(self):
        r"""return other_args.
               """
        return self._config_file


class ConfigParser(object):
    """
    use to parse the user defined parameters and use these to modify the
    pipeline's parameter setting.
    值得注意的是，目前各阶段的参数是放置于同一个 dict 中的，因此需要编程时保证命名空间不冲突。
    config 优先级：命令行 > config file > default config
    """

    def __init__(self, task, model, dataset, config_file=None,
                 saved_model=True, train=True, other_args=None, hyper_config_dict=None):
        """
        Args:
            task, model, dataset (str): 用户在命令行必须指明的三个参数
            config_file (str): 配置文件的文件名，将在项目根目录下进行搜索
            other_args (dict): 通过命令行传入的其他参数
        """
        self.config = {}
        self._parse_external_config(task, model, dataset, saved_model, train, other_args, hyper_config_dict)
        self._parse_config_file(config_file)
        self._load_default_config()
        self._init_device()

    def _parse_external_config(self, task, model, dataset,
                               saved_model=True, train=True, other_args=None, hyper_config_dict=None):
        if task is None:
            raise ValueError('the parameter task should not be None!')
        if model is None:
            raise ValueError('the parameter model should not be None!')
        if dataset is None:
            raise ValueError('the parameter dataset should not be None!')
        # 目前暂定这三个参数必须由用户指定
        self.config['task'] = task
        self.config['model'] = model
        self.config['dataset'] = dataset
        self.config['saved_model'] = saved_model
        self.config['train'] = False if task == 'map_matching' else train
        if other_args is not None:
            # TODO: 这里可以设计加入参数检查，哪些参数是允许用户通过命令行修改的
            for key in other_args:
                self.config[key] = other_args[key]
        if hyper_config_dict is not None:
            # 超参数调整时传入的待调整的参数，优先级低于命令行参数
            for key in hyper_config_dict:
                self.config[key] = hyper_config_dict[key]

    def _parse_config_file(self, config_file):
        if config_file is not None:
            # TODO: 对 config file 的格式进行检查
            if os.path.exists('./{}.json'.format(config_file)):
                with open('./{}.json'.format(config_file), 'r') as f:
                    x = json.load(f)
                    for key in x:
                        if key not in self.config:
                            self.config[key] = x[key]
            else:
                raise FileNotFoundError(
                    'Config file {}.json is not found. Please ensure \
                    the config file is in the root dir and is a JSON \
                    file.'.format(config_file))

    def _load_default_config(self):
        # 首先加载 task config
        with open('./libcity/config/task_config.json', 'r') as f:
            task_config = json.load(f)
            if self.config['task'] not in task_config:
                raise ValueError(
                    'task {} is not supported.'.format(self.config['task']))
            task_config = task_config[self.config['task']]
            # check model and dataset
            if self.config['model'] not in task_config['allowed_model']:
                raise ValueError('task {} do not support model {}'.format(
                    self.config['task'], self.config['model']))
            model = self.config['model']
            # 加载 dataset、executor、evaluator 的模块
            if 'dataset_class' not in self.config:
                self.config['dataset_class'] = task_config[model]['dataset_class']
            if self.config['task'] == 'traj_loc_pred' and 'traj_encoder' not in self.config:
                self.config['traj_encoder'] = task_config[model]['traj_encoder']
            if self.config['task'] == 'eta' and 'eta_encoder' not in self.config:
                self.config['eta_encoder'] = task_config[model]['eta_encoder']
            if 'executor' not in self.config:
                self.config['executor'] = task_config[model]['executor']
            if 'evaluator' not in self.config:
                self.config['evaluator'] = task_config[model]['evaluator']
            # 对于 LSTM RNN GRU 使用的都是同一个类，只是 RNN 模块不一样而已，这里做一下修改
            if self.config['model'].upper() in ['LSTM', 'GRU', 'RNN']:
                self.config['rnn_type'] = self.config['model']
                self.config['model'] = 'RNN'
            # if self.config['dataset'] not in task_config['allowed_dataset']:
            #     raise ValueError('task {} do not support dataset {}'.format(
            #         self.config['task'], self.config['dataset']))
        # 接着加载每个阶段的 default config
        default_file_list = []
        # model
        default_file_list.append('model/{}/{}.json'.format(self.config['task'], self.config['model']))
        # dataset
        default_file_list.append('data/{}.json'.format(self.config['dataset_class']))
        # executor
        default_file_list.append('executor/{}.json'.format(self.config['executor']))
        # evaluator
        default_file_list.append('evaluator/{}.json'.format(self.config['evaluator']))
        # 加载所有默认配置
        for file_name in default_file_list:
            with open('./libcity/config/{}'.format(file_name), 'r') as f:
                x = json.load(f)
                for key in x:
                    if key not in self.config:
                        self.config[key] = x[key]
        # 加载数据集config.json
        with open('./raw_data/{}/config.json'.format(self.config['dataset']), 'r') as f:
            x = json.load(f)
            for key in x:
                if key == 'info':
                    for ik in x[key]:
                        if ik not in self.config:
                            self.config[ik] = x[key][ik]
                else:
                    if key not in self.config:
                        self.config[key] = x[key]

    def _init_device(self):
        use_gpu = self.config.get('gpu', True)
        gpu_id = self.config.get('gpu_id', 0)
        if use_gpu:
            torch.cuda.set_device(gpu_id)
        self.config['device'] = torch.device(
            "cuda:%d" % gpu_id if torch.cuda.is_available() and use_gpu else "cpu")

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getitem__(self, key):
        if key in self.config:
            return self.config[key]
        else:
            raise KeyError('{} is not in the config'.format(key))

    def __setitem__(self, key, value):
        self.config[key] = value

    def __contains__(self, key):
        return key in self.config

    # 支持迭代操作
    def __iter__(self):
        return self.config.__iter__()

