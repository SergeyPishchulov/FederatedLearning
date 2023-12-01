from typing import List

from datasets import Data
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


class FederatedMLTask:
    def __init__(self, node_cnt, conf: FederatedMLTaskConfiguration, random_seed, args):
        self.args = args
        self.data = Data(conf.dataset, node_cnt, 0, random_seed)  # TODO dumb
        self.conf = conf
        self.node_cnt = node_cnt
        self.done = False
        # Data-size-based aggregation weights
        sample_size = []
        for i in range(node_cnt):
            sample_size.append(len(self.data.train_loader[i]))
        self.size_weights = [x / sum(sample_size) for x in sample_size]
        self.central_node = Node(-1, self.data.test_loader[0], self.data.test_set, self.args, self.node_cnt)
        self.client_nodes = {}
        for i in range(self.node_cnt):
            self.client_nodes[i] = Node(i, self.data.train_loader[i], self.data.train_set, self.args, self.node_cnt)


class Client:
    def __init__(self, hub, node_by_ft, args):
        # self.nodes:List[Node] = nodes#TODO delete?
        self.hub = hub
        self.args = args
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
        if 'fedlaw' in self.args.server_method:
            node.model.load_param(copy.deepcopy(central_node.model.get_param(clone=True)))
        else:
            node.model.load_state_dict(copy.deepcopy(central_node.model.state_dict()))
        epoch_losses = []
        if self.args.client_method == 'local_train':
            for epoch in range(self.args.E):
                loss = self.client_localTrain(self.args, node)  # TODO check if not working
                epoch_losses.append(loss)
            mean_loss = sum(epoch_losses) / len(epoch_losses)
        else:
            raise NotImplemented('Still only local_train =(')
        acc = validate(self.args, node)
        node.epochs_performed += 1
        return mean_loss, acc


class Hub:
    def __init__(self):
        pass

    def receive_server_model(self, ft):
        return ft.central_node


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
    hub = Hub()
    clients = [Client(hub, {ft: x}, ft.args)
               for x in ft.client_nodes.values()]

    final_test_acc_recorder = RunningAverage()
    test_acc_recorder = []

    while not ft.done:
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

    print(ft.args.server_method + ft.args.client_method + ', final_testacc is ', final_test_acc_recorder.value())
