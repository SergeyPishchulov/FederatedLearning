from typing import List

from datasets import Data
from nodes import Node
from args import args_parser
from utils import *
from server_funct import *
from client_funct import *
import os
from copy import deepcopy


class FederatedMLTaskConfiguration:
    def __init__(self, dataset, nn_architecture, epochs):
        self.dataset = dataset
        self.nn_architecture = nn_architecture
        self.epochs = epochs
        self.iid = 1  # data is iid


# TODO подумать. Копировать args, заменять в них датасет, архитектуру и не париться
class FederatedMLTask:
    def __init__(self, node_cnt, conf: FederatedMLTaskConfiguration, random_seed, args):
        self.args = args
        self.data = Data(conf.dataset, node_cnt, 0, random_seed)  # TODO dumb
        self.conf = conf
        self.node_cnt = node_cnt
        # Data-size-based aggregation weights
        sample_size = []
        for i in range(node_cnt):
            sample_size.append(len(self.data.train_loader[i]))
        self.size_weights = [x / sum(sample_size) for x in sample_size]
        self.central_node: Node = None
        self.client_nodes: List[Node] = None
        self.init_nodes()

    def init_nodes(self):
        self.central_node = Node(-1, self.data.test_loader[0], self.data.test_set, self.args, self.node_cnt)
        ft.client_nodes = {}
        for i in range(self.node_cnt):
            self.client_nodes[i] = Node(i, self.data.train_loader[i], self.data.train_set, self.args, self.node_cnt)


if __name__ == '__main__':
    user_args = args_parser()
    setup_seed(user_args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = user_args.device
    torch.cuda.set_device('cuda:' + user_args.device)

    fedeareted_tasks_configs = []
    fedeareted_tasks_configs.append(FederatedMLTaskConfiguration(
        dataset='cifar10',
        nn_architecture='ResNet20',
        epochs='2',
    ))
    fedeareted_tasks_configs.append(FederatedMLTaskConfiguration(
        dataset='cifar10',
        nn_architecture='ResNet56',
        epochs='1'
    ))
    node_num = 5
    random_seed = 10
    conf = fedeareted_tasks_configs[0]
    ft = FederatedMLTask(node_num, conf, random_seed, args=deepcopy(user_args))

    # Start the FL training
    final_test_acc_recorder = RunningAverage()
    test_acc_recorder = []
    for rounds in range(ft.args.T):
        print('===============Stage 1 The {:d}-th round==============='.format(rounds + 1))
        lr_scheduler(rounds, ft.client_nodes, ft.args)
        # Client update
        ft.client_nodes, train_loss = Client_update(ft.args, ft.client_nodes, ft.central_node)
        avg_client_acc = Client_validate(ft.args, ft.client_nodes)
        print(ft.args.server_method + ft.args.client_method + ', averaged clients personalization acc is ', avg_client_acc)

        # Partial select function
        if ft.args.select_ratio == 1.0:
            select_list = [idx for idx in range(len(ft.client_nodes))]
        else:
            select_list = generate_selectlist(ft.client_nodes, ft.args.select_ratio)

        # Server update
        ft.central_node = Server_update(ft.args, ft.central_node, ft.client_nodes, select_list, ft.size_weights)
        acc = validate(ft.args, ft.central_node, which_dataset='local')
        print(ft.args.server_method + ft.args.client_method + ', global model test acc is ', acc)
        test_acc_recorder.append(acc)

        # Final acc recorder
        if rounds >= ft.args.T - 10:
            final_test_acc_recorder.update(acc)

    print(ft.args.server_method + ft.args.client_method + ', final_testacc is ', final_test_acc_recorder.value())
