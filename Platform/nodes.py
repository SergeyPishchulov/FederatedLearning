import copy
import numpy as np
import torch
from torch.utils.data import DataLoader

from fl_log import logging_print
from datasets import DatasetSplit, DatasetPartiallyAvailable
from utils import init_model
from utils import init_optimizer, model_parameter_vector


class Node(object):
    def __init__(self, num_id, local_data, train_set, args, ft_id):
        self.num_id = num_id
        self.args = args
        self._local_data = local_data
        self._train_set = train_set
        self.sample_size = len(local_data)
        logging_print(f"--- --- DATA for node {self.num_id} task {ft_id} is {self.sample_size}")
        self.deadline_by_round = None  # max time to perform round №r
        if num_id == -1:
            self.valid_ratio = args.server_valid_ratio
        else:
            self.valid_ratio = args.client_valid_ratio

        if self.args.dataset == 'cifar10' or self.args.dataset == 'fmnist':
            self.num_classes = 10
        elif self.args.dataset == 'cifar100':
            self.num_classes = 100

        if num_id == -1:
            self.local_data, self.validate_set = self.train_val_split_forServer(self._local_data.indices,
                                                                                self._train_set,
                                                                                self.valid_ratio, self.num_classes,
                                                                                input_tss=None)
        # if args.iid == 1 or num_id == -1:
        #     pass  # TODO [sp] even with iid arg we use method for non-iid
        #     # for the server, use the validate_set as the training data, and use local_data for testing
        #     self.local_data, self.validate_set = self.train_val_split_forServer(local_data.indices, train_set,
        #                                                                         self.valid_ratio, self.num_classes)
        # else:
        #     pass
        #     # self.local_data, self.validate_set = self.train_val_split(local_data, train_set, self.valid_ratio)

        self.model = init_model(self.args.local_model, self.args)
        if not self.args.debug:
            self.model = self.model.cuda()
        self.optimizer = init_optimizer(self.num_id, self.model, args)

        # node init for fedadam's server
        if args.server_method == 'fedadam' and num_id == -1:
            m = copy.deepcopy(self.model)
            self.zero_weights(m)
            self.m = m
            v = copy.deepcopy(self.model)
            self.zero_weights(v)
            self.v = v

    def data_for_round_is_available(self, round_num):
        if isinstance(self.ds_train, DatasetPartiallyAvailable):
            return self.ds_train.get_parts_available() >= (round_num + 1)
        return True

    def set_datasets(self, input_tss):
        # for the server, use the validate_set as the training data, and use local_data for testing
        # TODO [sp] even with iid=1 we use method for non-iid
        if isinstance(self._local_data, torch.utils.data.dataset.Subset):
            ld = self._local_data.indices
        else:
            ld = self._local_data
        self.local_data, self.validate_set = self.train_val_split(ld, self._train_set,
                                                                  self.valid_ratio,
                                                                  input_tss)

    def zero_weights(self, model):
        for n, p in model.named_parameters():
            p.data.zero_()

    def train_val_split(self, idxs, train_set, valid_ratio, input_tss=None):

        np.random.shuffle(idxs)

        validate_size = valid_ratio * len(idxs)

        idxs_test = idxs[:int(validate_size)]
        idxs_train = idxs[int(validate_size):]

        ds_train = DatasetSplit(train_set, idxs_train)
        if input_tss is not None:
            ds_train = DatasetPartiallyAvailable(ds_train, input_tss)
        self.ds_train = ds_train
        train_loader = DataLoader(ds_train,
                                  batch_size=self.args.batchsize, num_workers=0, shuffle=True)

        test_loader = DataLoader(DatasetSplit(train_set, idxs_test),
                                 batch_size=self.args.validate_batchsize, num_workers=0, shuffle=True)

        return train_loader, test_loader

    def train_val_split_forServer(self, idxs, train_set, valid_ratio, num_classes=10, input_tss=None):  # local data index, trainset

        np.random.shuffle(idxs)

        validate_size = int(valid_ratio * len(idxs))

        # generate proxy dataset with balanced classes
        idxs_test = []

        if self.args.longtail_proxyset == 'none':
            test_class_count = [int(validate_size) / num_classes for _ in range(num_classes)]
        elif self.args.longtail_proxyset == 'LT':
            imb_factor = 0.1
            test_class_count = [int(validate_size / num_classes * (imb_factor ** (_classes_idx / (num_classes - 1.0))))
                                for _classes_idx in range(num_classes)]

        k = 0
        while sum(test_class_count) != 0:
            if test_class_count[train_set[idxs[k]][1]] > 0:
                idxs_test.append(idxs[k])
                test_class_count[train_set[idxs[k]][1]] -= 1
            else:
                pass
            k += 1
        label_list = []
        for k in idxs_test:
            label_list.append(train_set[k][1])

        idxs_train = [idx for idx in idxs if idx not in idxs_test]

        ds_train = DatasetSplit(train_set, idxs_train)
        if input_tss is not None:
            ds_train = DatasetPartiallyAvailable(ds_train, input_tss)
        self.ds_train = ds_train

        train_loader = DataLoader(ds_train,
                                  batch_size=self.args.batchsize, num_workers=0, shuffle=True)
        test_loader = DataLoader(DatasetSplit(train_set, idxs_test),
                                 batch_size=self.args.validate_batchsize, num_workers=0, shuffle=True)

        return train_loader, test_loader


# Tools for long-tailed functions
def label_indices2indices(list_label2indices):
    indices_res = []
    for indices in list_label2indices:
        indices_res.extend(indices)

    return indices_res


def _get_img_num_per_cls(list_label2indices_train, num_classes, imb_factor, imb_type):
    img_max = len(list_label2indices_train) / num_classes
    img_num_per_cls = []
    if imb_type == 'exp':
        for _classes_idx in range(num_classes):
            num = img_max * (imb_factor ** (_classes_idx / (num_classes - 1.0)))
            img_num_per_cls.append(int(num))

    return img_num_per_cls


def train_long_tail(list_label2indices_train, num_classes, imb_factor, imb_type):
    new_list_label2indices_train = label_indices2indices(copy.deepcopy(list_label2indices_train))
    img_num_list = _get_img_num_per_cls(copy.deepcopy(new_list_label2indices_train), num_classes, imb_factor, imb_type)
    print('img_num_class')
    print(img_num_list)

    list_clients_indices = []
    classes = list(range(num_classes))
    for _class, _img_num in zip(classes, img_num_list):
        indices = list_label2indices_train[_class]
        np.random.shuffle(indices)
        idx = indices[:_img_num]
        list_clients_indices.append(idx)
    num_list_clients_indices = label_indices2indices(list_clients_indices)
    print('All num_data_train')
    print(len(num_list_clients_indices))

    return img_num_list, list_clients_indices
