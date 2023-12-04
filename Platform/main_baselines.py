from typing import List, Optional

import torch.nn

from message import MessageToClient, MessageToHub
from statistics import Statistics
from experiment_config import get_configs
from datasets import Data
from args import args_parser
from utils import *
from server_funct import *
from client_funct import *
import os
from copy import deepcopy


class FederatedMLTask:
    def __init__(self, id, args):
        self.args = args
        self.data = Data(args)  # TODO dumb
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
        self.client_nodes = [Node(i, self.data.train_loader[i],
                                  self.data.train_set, self.args, self.node_cnt)
                             for i in range(self.node_cnt)]



class Client:
    def __init__(self, id, node_by_ft_id, args_by_ft_id, agr_model_by_ft_id_round):
        self.id = id  # TODO set pipe
        self.plan = self._get_plan()
        # self.hub = hub#temporary. instead of pipe
        # self.args = args
        self.node_by_ft_id = node_by_ft_id
        self.args_by_ft_id = args_by_ft_id
        self.agr_model_by_ft_id_round = agr_model_by_ft_id_round

    def _get_plan(self):
        return combine_lists([
            [(round, task.id) for round in range(ROUNDS)] for task in tasks
        ])

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

    def run(self):
        for r, ft_id in self.plan:
            if (ft_id, r - 1) in self.agr_model_by_ft_id_round:
                agr_model = self.agr_model_by_ft_id_round[(ft_id, r - 1)]
                ft_args = self.args_by_ft_id[ft_id]
                node = self.node_by_ft_id[ft_id]  # TODO delete hub when set pipe in __init__
                if 'fedlaw' in ft_args.server_method:
                    node.model.load_param(copy.deepcopy(
                        agr_model.get_param(clone=True)))
                else:
                    node.model.load_state_dict(copy.deepcopy(
                        agr_model.state_dict()))
                epoch_losses = []
                if ft_args.client_method == 'local_train':
                    for epoch in range(ft_args.E):
                        loss = self.client_localTrain(ft_args, node)  # TODO check if not working
                        epoch_losses.append(loss)
                    mean_loss = sum(epoch_losses) / len(epoch_losses)
                else:
                    raise NotImplemented('Still only local_train =(')
                acc = validate(ft_args, node)
                node.rounds_performed += 1  # TODO not to mess with r
                response = MessageToHub(node.rounds_performed - 1, ft_id,
                                        acc, mean_loss, node.model, self.id)
                yield response
            else:
                raise ValueError(f"Agr model from prev step is not found {self.agr_model_by_ft_id_round.keys()}")


# def perform_one_round(self, mes: MessageToClient, hub):
#     ft_args = self.args_by_ft_id[mes.ft_id]
#     node = self.node_by_ft_id[mes.ft_id]  # TODO delete hub when set pipe in __init__
#     # central_node = #hub.receive_server_model(mes.ft_id)
#     if 'fedlaw' in ft_args.server_method:
#         node.model.load_param(copy.deepcopy(
#             mes.agr_model.get_param(clone=True)))
#     else:
#         node.model.load_state_dict(copy.deepcopy(
#             mes.agr_model.state_dict()))
#     epoch_losses = []
#     if ft_args.client_method == 'local_train':
#         for epoch in range(ft_args.E):
#             loss = self.client_localTrain(ft_args, node)  # TODO check if not working
#             epoch_losses.append(loss)
#         mean_loss = sum(epoch_losses) / len(epoch_losses)
#     else:
#         raise NotImplemented('Still only local_train =(')
#     acc = validate(ft_args, node)
#     node.rounds_performed += 1
#     response = MessageToHub(node.rounds_performed, mes.ft_id,
#                             acc, mean_loss, node.model)
#     return response

class TrainingJournal:
    def __init__(self, task_ids):
        self.d = {}  # key is (ft_id, client_id, round); value is model
        self.latest_aggregated_round = {i: -1 for i in task_ids}

    def mark_as_aggregated(self, ft_id):
        self.latest_aggregated_round[ft_id] += 1
        # TODO bug if we skip some rounds

    def save_local(self, ft_id, client_id, round_num, model):
        if (ft_id, client_id, round_num) not in self.d:
            self.d[(ft_id, client_id, round_num)] = model
        else:
            raise KeyError("Key already exists")

    def get_ft_to_aggregate(self, client_ids):
        for ft_id, latest_round in self.latest_aggregated_round.items():
            # print(f'Searching ({ft_id},_,{latest_round+1}) in keys. client_ids is {client_ids}')
            if all((ft_id, cl_id, latest_round + 1) in self.d
                   for cl_id in client_ids):
                return ft_id, latest_round + 1
        raise ValueError(f"No task to aggregate. d keys: {self.d.keys()}")
        return None, None


class Hub:
    def __init__(self, tasks: List[FederatedMLTask], clients, args):
        self.tasks = tasks
        self.clients = clients
        self.stat = Statistics(tasks, clients, args)
        self.journal = TrainingJournal([ft.id for ft in tasks])

    def receive_server_model(self, ft_id):
        return self.tasks[ft_id].central_node

    def get_select_list(self, ft, client_ids):
        if ft.args.select_ratio == 1.0:
            select_list = client_ids
        else:
            select_list = generate_selectlist(client_ids, ft.args.select_ratio)
        return select_list


if __name__ == '__main__':
    user_args = args_parser()
    setup_seed(user_args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = user_args.device
    torch.cuda.set_device('cuda:' + user_args.device)

    fedeareted_tasks_configs = get_configs(user_args)
    tasks = [FederatedMLTask(id, c) for id, c in enumerate(fedeareted_tasks_configs)]

    ROUNDS = 10
    clients = []
    for client_id in range(user_args.node_num):
        clients.append(Client(client_id, {ft.id: ft.client_nodes[client_id] for ft in tasks},
                              args_by_ft_id={ft.id: ft.args for ft in tasks},
                              agr_model_by_ft_id_round={(ft.id, -1): ft.central_node.model for ft in tasks}))
    hub = Hub(tasks, clients, user_args)
    final_test_acc_recorder = RunningAverage()
    test_acc_recorder = []

    gens = [c.run() for c in clients]
    for responses in zip(*gens):
        r: MessageToHub
        for r in responses:
            hub.journal.save_local(r.ft_id, r.client_id, r.round_num, r.model)
            hub.stat.save_client_ac(r.client_id, r.ft_id, r.round_num, r.acc)

        next_ft_id, ag_round = hub.journal.get_ft_to_aggregate([c.id for c in clients])
        if next_ft_id is not None:
            ft = tasks[next_ft_id]
            Server_update(ft.args, ft.central_node.model, [n.model for n in ft.client_nodes],
                          hub.get_select_list(ft, [c.id for c in clients]),  # TODO note that local models are took from nodes, not from journal
                          ft.size_weights)
            hub.journal.mark_as_aggregated(ft.id)
            for c in clients:  # TODO make through pipe
                c.agr_model_by_ft_id_round[(ft.id, ag_round)] = ft.central_node.model
            acc = validate(ft.args, ft.central_node, which_dataset='local')
            hub.stat.save_agr_ac(ft.id,
                                 round=ag_round,
                                 acc=acc)
        hub.stat.to_csv()
        hub.stat.plot_accuracy()
        # TODO delete client_nodes from ft.
