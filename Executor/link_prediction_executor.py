import os
import time
import torch

from mtlib.nn import CAWN
from .abstract_executor import AbstractExecutor

class Link_Prediction(AbstractExecutor):
    def __init__(self, config, model, data_feature):
        super(AbstractExecutor, self).__init__(config, model, data_feature)
        self.model=model
        self.config=config
        self.device=self.config.get('device', torch.device('cpu'))
        self.model = model.to(self.device)
        self.exp_id = self.config.get('exp_id', None)
        self.epochs = self.config.get('max_epoch', 100)
        self.learner = self.config.get('learner', 'adam')
        self.learning_rate = self.config.get('learning_rate', 0.01)

        self._logger = getLogger()

        self.weight_decay = self.config.get('weight_decay', 0)#自此以下自己生成
        self.lr_beta1 = self.config.get('lr_beta1', 0.9)
        self.lr_beta2 = self.config.get('lr_beta2', 0.999)
        self.lr_betas = (self.lr_beta1, self.lr_beta2)
        self.lr_alpha = self.config.get('lr_alpha', 0.99)
        self.lr_epsilon = self.config.get('lr_epsilon', 1e-8)
        self.lr_momentum = self.config.get('lr_momentum', 0)
        self.lr_decay = self.config.get('lr_decay', False)
        self.lr_scheduler_type = self.config.get('lr_scheduler', 'multisteplr')
        self.lr_decay_ratio = self.config.get('lr_decay_ratio', 0.1)
        self.milestones = self.config.get('steps', [])
        self.step_size = self.config.get('step_size', 10)
        self.lr_lambda = self.config.get('lr_lambda', lambda x: x)
        self.lr_T_max = self.config.get('lr_T_max', 30)
        self.lr_eta_min = self.config.get('lr_eta_min', 0)
        self.lr_patience = self.config.get('lr_patience', 10)
        self.lr_threshold = self.config.get('lr_threshold', 1e-4)

        self.optimizer = self._build_optimizer()

    def evaluate(self, node_reps,interactions):
        """
        use model to test data
        """
        acc = []
        ap = []
        auc = []
        size = len(interactions[:, 0])
        head_index = interactions[:, 0].astype(numpy.int)
        tail_index = interactions[:, 1].astype(numpy.int)
        train_sample = RandEdgeSampler(head_index, tail_index)
        _, neg_index = train_sample.sample(size)
        src_embed = node_reps[head_index]
        dst_embed = node_reps[tail_index]
        neg_embed = node_reps[neg_index]
        pos_score = numpy.sum(src_embed * dst_embed, axis=1)
        neg_score = numpy.sum(src_embed * neg_embed, axis=1)

        pred_score = numpy.concatenate([pos_score, neg_score])
        pred_label = pred_score > 0.5
        true_label = numpy.concatenate([numpy.ones(size), numpy.zeros(size)])
        acc.append((pred_label == true_label).mean())
        ap.append(average_precision_score(true_label, pred_score))
        # f1.append(f1_score(true_label, pred_label))
        auc.append(roc_auc_score(true_label, pred_score))
        return eval_data

    def train(self, train_dataloader,valid_data):
        """
        use data to train model with config
        """
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = 0
        train_time = []
        eval_time = []

        for epoch_idx in range(self._epoch_num, self.epochs):
            start_time = time.time()
            self.model.train()
            train_loss = self._train_epoch(train_dataloader, epoch_idx, self.loss_func)
            t1 = time.time()
            train_time.append(t1 - start_time)
            self._logger.info("epoch complete!")
            self._logger.info("evaluating now!")
            t2 = time.time()
            val_loss = self._valid_epoch(valid_data, epoch_idx, self.loss_func)
            end_time = time.time()
            eval_time.append(end_time - t2)

            if self.lr_scheduler is not None:
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(val_loss)
                else:
                    self.lr_scheduler.step()

            if (epoch_idx % self.log_every) == 0:
                log_lr = self.optimizer.param_groups[0]['lr']
                message = 'Epoch [{}/{}] train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s'\
                    .format(epoch_idx, self.epochs, train_loss, val_loss, log_lr, (end_time - start_time))
                self._logger.info(message)

            if self.hyper_tune:
                # use ray tune to checkpoint
                with tune.checkpoint_dir(step=epoch_idx) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    self.save_model(path)
                # ray tune use loss to determine which params are best
                tune.report(loss=val_loss)

            if val_loss < min_val_loss:
                wait = 0
                if self.saved:
                    model_file_name = self.save_model_with_epoch(epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                      'saving to {}'.format(min_val_loss, val_loss, model_file_name))
                min_val_loss = val_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break
        if len(train_time) > 0:
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), sum(train_time) / len(train_time),
                                     sum(eval_time) / len(eval_time)))
        if self.load_best_epoch:
            self.load_model_with_epoch(best_epoch)
            save_list = os.listdir(self.cache_dir)
            for save_file in save_list:
                if '.tar' in save_file:
                    os.remove(self.cache_dir + '/' + save_file)
        return min_val_loss

    def _train_epoch(self, train_dataloader, epoch_idx, loss_func=None):
        """
        完成模型一个轮次的训练

        Returns:
            float: 训练集的损失值
        """
        for idx,data in enumerate(train_dataloader):
            loss=self.model.loss(data)
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def _valid_epoch(self, valid_data, epoch_idx, loss_func=None):
        """
        完成模型一个轮次的评估

        Args:
            eval_dataloader: 评估数据
            epoch_idx: 轮次数
            loss_func: 损失函数

        Returns:
            float: 验证集的损失值
        """

        with torch.no_grad():
            self.model.eval()
            node_reps = self.model.emb()
            valid_evl = self.evaluate(node_reps,valid_data,loss_func)
            loss_func = loss_func if loss_func is not None else self.model.calculate_loss
            loss = loss_func({'node_features': node_features, 'node_labels': node_labels, 'mask': valid_mask})
            self._writer.add_scalar('eval loss', loss, epoch_idx)
            return loss.item()


    def _build_optimizer(self):
        """
        根据全局参数`learner`选择optimizer
        """
        self._logger.info('You select `{}` optimizer.'.format(self.learner.lower()))
        if self.learner.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, betas=self.lr_betas, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                        momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate,
                                            eps=self.lr_epsilon, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate,
                                            alpha=self.lr_alpha, eps=self.lr_epsilon,
                                            momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=self.learning_rate,
                                               eps=self.lr_epsilon, betas=self.lr_betas)
        else:
            self._logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, weight_decay=self.weight_decay)
        return optimizer