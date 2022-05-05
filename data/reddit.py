import ssl
import urllib
import numpy
import os

from .utils import TqdmUpTo
import numpy as np
import torch

from .dy_dataset  import DGBuiltinDataset

class RedditDataset(DGBuiltinDataset):
    def __init__(self, raw_dir=None, force_reload=False, verbose=False):
        name="reddit"
        _url='https://snap.stanford.edu/jodie/{}.csv'.format(name)
        super(RedditDataset, self).__init__(name=name,
                                            url=_url,
                                            raw_dir=raw_dir,
                                            force_reload=force_reload,
                                            verbose=verbose)

    def process(self):
        src_dst_timePath = os.path.join(self.processed_path, '{}_src_dst_time.npy'.format(self.name))
        edge_featuresPath = os.path.join(self.processed_path, '{}_edge_features.npy'.format(self.name))
        node_featuresPath = os.path.join(self.processed_path, '{}_node_features.npy'.format(self.name))
        node_statePath = os.path.join(self.processed_path,'{}_node_state.npy'.format(self.name))

        if not os.path.exists(src_dst_timePath) and os.path.exists(edge_featuresPath) and os.path.exists\
                    (node_featuresPath) and os.path.exists(node_statePath) :
            print("Start Process Data ...")
            print("starting process raw data in {}".format(self.name))
            PATH = os.path.join(self.raw_dir, '{}.csv'.format(self.name))

            u_list, i_list, ts_list, label_list = [], [], [], []
            feat_l = []
            idx_list = []

            with open(PATH) as f:
                s = next(f)
                print(s)
                for idx, line in enumerate(f):
                    e = line.strip().split(',')
                    u = int(e[0])
                    i = int(e[1])

                    ts = int(float(e[2]))
                    label = int(e[3])
                    feat = numpy.array([float(x) for x in e[4:]])

                    u_list.append(u)
                    i_list.append(i)
                    ts_list.append(ts)
                    label_list.append(label)
                    idx_list.append(idx)

                    feat_l.append(feat)
            src = numpy.array(u_list)[:, numpy.newaxis]
            dst = numpy.array(i_list)[:, numpy.newaxis]
            time = numpy.array(ts_list)[:, numpy.newaxis]
            idx_f = numpy.array(idx_list)[:, numpy.newaxis]
            label_f = numpy.array(label_list)[:, numpy.newaxis]

            src_dst_time = numpy.concatenate((idx_f, src, dst, time), axis=1)
            idx_label = numpy.hstack((idx_f, label_f))
            edge_features = numpy.array(feat_l)

            assert (src_dst_time[:, 1].max() - src_dst_time[:, 1].min() + 1 == len(list(numpy.unique(src_dst_time[:, 1]))))
            assert (src_dst_time[:, 2].max() - src_dst_time[:, 2].min() + 1 == len(list(numpy.unique(src_dst_time[:, 2]))))

            upper_u = src_dst_time[:, 1].max() + 1
            src_dst_time[:, 2] = src_dst_time[:, 2] + upper_u

            print(edge_features.shape)
            # empty = numpy.zeros(feat.shape[1])[numpy.newaxis, :]
            # edge_features = numpy.vstack([empty, feat])

            max_idx = src_dst_time[:, 2].max()
            rand_feat = numpy.zeros((max_idx + 1, edge_features.shape[1]))
            # print(feat.shape)
            os.makedirs(self.processed_path)
            numpy.save(edge_featuresPath, edge_features)
            numpy.save(node_featuresPath, rand_feat)
            numpy.save(src_dst_timePath, src_dst_time)
            numpy.save(node_statePath, idx_label)

        src_dst_time = np.load(src_dst_timePath)
        self._edge_index = torch.LongTensor(src_dst_time[0:2, :])
        self._timestamp = torch.FloatTensor(src_dst_time[:, -1])

        edge_features = np.load(edge_featuresPath)
        self._edge_features = torch.FloatTensor(edge_features)

        node_features = np.load(node_featuresPath)
        self._node_features = torch.FloatTensor(node_features)

        node_state = np.load(node_statePath)
        self._node_state = node_state


    def download(self):
        print("Start Downloading File....")
        file=self.url.split("/")([-1])
        if os.path.exists(self.raw_path):
            os.mkdir(self.raw_path)
        with TqdmUpTo(unit='B',unit_scale=True,unit_divisor=1024,miniters=1,desc=file) as t:
            urllib.request.urlretrieve(self.url,os.path.join(self.raw_path, 'reddit.csv'),t.updata_to)
