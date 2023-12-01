from typing import List

from experiment_config import get_configs
from datasets import Data
from args import args_parser
from utils import *
from server_funct import *
from client_funct import *
import os
from copy import deepcopy

from itertools import zip_longest


def combine_lists(*l):
    return [j for i in zip_longest(*l) for j in i if j]


class FederatedMLTask:
    def __init__(self, id, args):
        self.args = args
        self.data = Data(conf.dataset, args)  # TODO dumb
        self.node_cnt = args.node_num
        self.id = id
        self.name = f'Task {id} {args.dataset} {args.local_model}'
        self.done = False
        # Data-size-based aggregation weights
        sample_size = []
        for i in range(self.node_cnt):
            sample_size.append(len(self.data.train_loader[i]))
        self.size_weights = [x / sum(sample_size) for x in sample_size]
        self.central_node = Node(-1, self.data.test_loader[0], self.data.test_set, self.args, self.node_cnt)
        self.client_nodes = {}
        for i in range(self.node_cnt):
            self.client_nodes[i] = Node(i, self.data.train_loader[i],
                                        self.data.train_set, self.args, self.node_cnt)


class Client:
    def __init__(self, hub, node_by_ft):
        self.hub = hub
        # self.args = args
        self.node_by_ft = node_by_ft

    def client_localTrain(self, args, node, loss=0.0):
        node.model.train()

        loss = 0.0
        train_loader = node.local_data  # iid
        for idx, (data, target) in enumerate(train_loader):
            # zero_grad
            node.optimizer.zero_grad()
            # train model
            data, target = data.cuda(), target.cuda()
            output_local = node.model(data)

            loss_local = F.cross_entropy(output_local, target)
            loss_local.backward()
            loss = loss + loss_local.item()
            node.optimizer.step()

        return loss / len(train_loader)

    def perform_one_round(self, ft: FederatedMLTask):
        node = self.node_by_ft[ft]
        central_node = self.hub.receive_server_model(ft)
        if 'fedlaw' in ft.args.server_method:
            node.model.load_param(copy.deepcopy(central_node.model.get_param(clone=True)))
        else:
            node.model.load_state_dict(copy.deepcopy(central_node.model.state_dict()))
        epoch_losses = []
        if ft.args.client_method == 'local_train':
            for epoch in range(ft.args.E):
                loss = self.client_localTrain(ft.args, node)  # TODO check if not working
                epoch_losses.append(loss)
            mean_loss = sum(epoch_losses) / len(epoch_losses)
        else:
            raise NotImplemented('Still only local_train =(')
        acc = validate(ft.args, node)
        node.epochs_performed += 1
        return mean_loss, acc


class Hub:
    def __init__(self):
        pass

    def receive_server_model(self, ft):
        return ft.central_node


if __name__ == '__main__':
    user_args = args_parser()
    # print(get_configs(user_args))
    # exit()
    setup_seed(user_args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = user_args.device
    torch.cuda.set_device('cuda:' + user_args.device)

    fedeareted_tasks_configs = get_configs(user_args)
    conf = fedeareted_tasks_configs[0]
    tasks = [FederatedMLTask(id, c) for id, c in enumerate(fedeareted_tasks_configs)]
    # ft = FederatedMLTask(args=fedeareted_tasks_configs[0])
    hub = Hub()
    clients = []
    for cl_num in range(user_args.node_num):
        clients.append(Client(hub, {ft: ft.client_nodes[cl_num] for ft in tasks}))
    # clients = [Client(hub, {ft: x}, ft.args)
    #            for ft in tasks for cn in ft.client_nodes.values()]

    final_test_acc_recorder = RunningAverage()
    test_acc_recorder = []

    plan = combine_lists([tasks[0]] * 2 + [tasks[1]] * 2)
    # while not all(ft.done for ft in tasks):
    for ft in plan:
        client_losses = []
        client_acc = []
        for c in clients:
            loss, acc = c.perform_one_round(ft)
            client_losses.append(loss)
            client_acc.append(acc)
        if all([c.node_by_ft[ft].epochs_performed == ft.args.T
                for c in clients]):
            ft.done = True
        train_loss = sum(client_losses) / len(client_losses)
        avg_client_acc = sum(client_acc) / len(client_acc)
        print(f"============= {ft.name} ============= ")
        print(ft.args.server_method + ft.args.client_method + ', averaged clients personalization acc is ',
              avg_client_acc)

        # Partial select function
        if ft.args.select_ratio == 1.0:
            select_list = [idx for idx in range(len(ft.client_nodes))]
        else:
            select_list = generate_selectlist(ft.client_nodes, ft.args.select_ratio)

        ft.central_node = Server_update(ft.args, ft.central_node, ft.client_nodes, select_list, ft.size_weights)
        acc = validate(ft.args, ft.central_node, which_dataset='local')
        print(ft.args.server_method + ft.args.client_method + ', global model test acc is ', acc)
        test_acc_recorder.append(acc)

    # print(ft.args.server_method + ft.args.client_method + ', final_testacc is ', final_test_acc_recorder.value())
