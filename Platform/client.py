import os
import time
import traceback

import torch

from message import MessageToHub, MessageToClient
from utils import validate, setup_seed, combine_lists
from utils import *
from server_funct import *
from client_funct import *


class Client:
    def __init__(self, id, node_by_ft_id, args_by_ft_id, agr_model_by_ft_id_round, user_args):
        self.id = id  # TODO set pipe
        # self.hub = hub#temporary. instead of pipe
        # self.args = args
        self.node_by_ft_id = node_by_ft_id
        self.args_by_ft_id = args_by_ft_id
        self.agr_model_by_ft_id_round = agr_model_by_ft_id_round
        self.user_args = user_args
        self.plan = self._get_plan()


    def _get_plan(self):
        rounds = self.user_args.T
        return combine_lists([
            [(round, ft_id) for round in range(rounds)] for ft_id in self.node_by_ft_id
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

    def _set_aggregated_model(self, ft_args, node, agr_model):
        if 'fedlaw' in ft_args.server_method:
            node.model.load_param(copy.deepcopy(
                agr_model.get_param(clone=True)))
        else:
            node.model.load_state_dict(copy.deepcopy(
                agr_model.state_dict()))

    def _train_one_round(self, ft_args, node):
        epoch_losses = []
        if ft_args.client_method == 'local_train':
            for epoch in range(ft_args.E):
                loss = self.client_localTrain(ft_args, node)  # TODO check if not working
                epoch_losses.append(loss)
            mean_loss = sum(epoch_losses) / len(epoch_losses)
            return mean_loss
        else:
            raise NotImplemented('Still only local_train =(')

    def setup(self):
        setup_seed(self.user_args.random_seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.user_args.device
        torch.cuda.set_device('cuda:' + self.user_args.device)

    def run(self, read_q, write_q):
        self.setup()
        while self.plan:
            r, ft_id = self.plan[0]
            while not read_q.empty():
                mes: MessageToClient = read_q.get()
                print(f'Client {self.id}: Got update form AGS for round {mes.round_num}, task {mes.ft_id}')
                self.agr_model_by_ft_id_round[(mes.ft_id, mes.round_num)] = mes.agr_model.clone()
                del mes.agr_model
                del mes

            if (ft_id, r - 1) in self.agr_model_by_ft_id_round:
                agr_model = self.agr_model_by_ft_id_round[(ft_id, r - 1)]
                ft_args = self.args_by_ft_id[ft_id]
                node = self.node_by_ft_id[ft_id]  # TODO delete hub when set pipe in __init__
                self._set_aggregated_model(ft_args, node, agr_model)
                mean_loss = self._train_one_round(ft_args, node)
                acc = validate(ft_args, node)
                node.rounds_performed += 1  # TODO not to mess with r
                response = MessageToHub(node.rounds_performed - 1, ft_id,
                                        acc, mean_loss,
                                        copy.deepcopy(node.model),
                                        self.id)
                try:
                    write_q.put(response)
                except Exception:
                    print(traceback.format_exc())
                self.plan.pop(0)
                print(f'Client {self.id} sent local model for round {response.round_num}, task {response.ft_id}')
            else:
                pass
                # print(
                #     f" Client {self.id}: Agr model from prev step is not found {self.agr_model_by_ft_id_round.keys()}")
                # time.sleep(1)
            # time.sleep(0.5)
        print(f'Client {self.id} is DONE')
