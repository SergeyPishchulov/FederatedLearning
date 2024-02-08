import argparse
import os
import time
import traceback
from datetime import datetime, timedelta
from pprint import pprint

import torch
from typing import Dict, List

from message import MessageToHub, MessageToClient, ResponseToHub, Period
from utils import validate, setup_seed, combine_lists
from utils import *
from server_funct import *
from client_funct import *


class LocalScheduler:
    def __init__(self, trained_ft_id_round):
        self.trained_ft_id_round: set = trained_ft_id_round

    def delete_from_plan(self, ft_id, r):
        """It notifies the scheduler that round is performed successfully, so it should not be scheduled again"""
        self.trained_ft_id_round.add((ft_id, r))


class MinDeadlineScheduler(LocalScheduler):
    def __init__(self, trained_ft_id_round):
        super().__init__(trained_ft_id_round)

    def get_next_task(self, agr_model_by_ft_id_round, node_by_ft_id: Dict[int, Node], rounds_cnt):
        ready = []
        planning_round_by_ft_id = {}
        for ft_id in node_by_ft_id:
            n: Node = node_by_ft_id[ft_id]
            last_aggregated_round = max(k[1] for k in agr_model_by_ft_id_round if k[0] == ft_id)
            planning_round = last_aggregated_round + 1
            if ((ft_id, planning_round) not in self.trained_ft_id_round and
                    n.data_for_round_is_available(planning_round)
                    and planning_round < rounds_cnt):
                ready.append((n.deadline_by_round[planning_round], ft_id))
                planning_round_by_ft_id[ft_id] = planning_round
        ready.sort()
        if not ready:
            return None, None
        deadline, ft_id = ready[0]
        return ft_id, planning_round_by_ft_id[ft_id]  # task with min deadline


class CyclicalScheduler(LocalScheduler):
    def __init__(self, trained_ft_id_round, user_args, node_by_ft_id):
        super().__init__(trained_ft_id_round)
        rounds = user_args.T
        self.plan = combine_lists([
            [(round, ft_id) for round in range(rounds)] for ft_id in node_by_ft_id
        ])

    def get_next_task(self, agr_model_by_ft_id_round, node_by_ft_id: Dict[int, Node], rounds_cnt):
        status = {}
        for (r, ft_id) in self.plan:
            n: Node = node_by_ft_id[ft_id]
            has_prev_model = (ft_id, r - 1) in agr_model_by_ft_id_round
            data_available = n.data_for_round_is_available(r)
            status[(r, ft_id)] = f"Data {int(data_available)}, prev_model {int(has_prev_model)}"
            if (has_prev_model and data_available):
                # print(f"    Client task is chosen {r, ft_id}")
                return ft_id, r
        # print(f"    Client plan is {self.plan}. Can not choose task. Status: ")
        # pprint(status)
        return None, None

    def delete_from_plan(self, ft_id, r):
        """It notifies the scheduler that round is performed successfully, so it should not be scheduled again"""
        super().delete_from_plan(ft_id, r)
        self.plan.remove((r, ft_id))


class Client:
    def __init__(self, id, node_by_ft_id, args_by_ft_id, agr_model_by_ft_id_round, user_args,
                 inter_ddl_periods_by_ft_id):
        self.id = id
        self.node_by_ft_id = node_by_ft_id
        self.args_by_ft_id = args_by_ft_id
        self.agr_model_by_ft_id_round = agr_model_by_ft_id_round
        self.user_args = user_args
        self.should_finish = False
        self.data_lens_by_ft_id: Dict[int, List] = {ft_id: [0] for ft_id in node_by_ft_id}
        self.trained_ft_id_round = set()
        self.scheduler = self.get_scheduler(user_args)
        self.inter_ddl_periods_by_ft_id = inter_ddl_periods_by_ft_id

    def get_scheduler(self, user_args):
        if user_args.local_scheduler == "CyclicalScheduler":
            print(f'    Client {self.id} SCHEDULER is set to CyclicalScheduler')
            return CyclicalScheduler(self.trained_ft_id_round, user_args, self.node_by_ft_id)
        elif user_args.local_scheduler == "MinDeadlineScheduler":
            print(f'    Client {self.id} SCHEDULER is set to MinDeadlineScheduler')
            return MinDeadlineScheduler(self.trained_ft_id_round)
        raise argparse.ArgumentError(user_args.local_scheduler, "Unknown value")

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

        return loss / len(train_loader), len(train_loader) * node.args.batchsize

    def _set_aggregated_model(self, ft_args, node, agr_model):
        if 'fedlaw' in ft_args.server_method:
            node.model.load_param(copy.deepcopy(
                agr_model.get_param(clone=True)))
        else:
            node.model.load_state_dict(copy.deepcopy(
                agr_model.state_dict()))

    def _train_one_round(self, ft_args, node):
        start_time = datetime.now()
        epoch_losses = []
        data_len = -1
        if ft_args.client_method == 'local_train':
            # print(f'Node {node.num_id} has available batches: {len(node.local_data)}')
            for epoch in range(ft_args.E):
                loss, data_len = self.client_localTrain(ft_args, node)  # TODO check if not working
                epoch_losses.append(loss)
            mean_loss = sum(epoch_losses) / len(epoch_losses)
            end_time = datetime.now()
            return mean_loss, data_len, start_time, end_time
        else:
            raise NotImplemented('Still only local_train =(')

    def setup(self):
        setup_seed(self.user_args.random_seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.user_args.device
        torch.cuda.set_device('cuda:' + self.user_args.device)

    def set_aggregated_model(self, ft_id, round_num, agr_model):
        local_model = copy.deepcopy(self.agr_model_by_ft_id_round[(ft_id, -1)])
        if 'fedlaw' in self.user_args.server_method:
            local_model.load_param(copy.deepcopy(agr_model.get_param(clone=True)))
        else:
            local_model.load_state_dict(copy.deepcopy(agr_model.state_dict()))
        self.agr_model_by_ft_id_round[(ft_id, round_num)] = local_model
        # self.agr_model_by_ft_id_round[(ft_id, round_num)] = copy.deepcopy(agr_model)#My version.BUG?

    def handle_messages(self, read_q, write_q):
        while not read_q.empty():
            mes: MessageToClient = read_q.get()
            # print(f'    Client {self.id}: Got update form AGS for round {mes.round_num}, task {mes.ft_id}')
            self.should_finish = mes.should_finish
            self.set_aggregated_model(mes.ft_id, mes.round_num, mes.agr_model)
            required_deadline = self.node_by_ft_id[mes.ft_id].deadline_by_round[mes.round_num]
            delay = max((datetime.now() - required_deadline), timedelta(seconds=0))
            write_q.put(ResponseToHub(self.id, mes.ft_id, mes.round_num, delay, final_message=mes.should_finish))
            del mes

    def set_deadlines(self):
        for ft_id, n in self.node_by_ft_id.items():
            n: Node
            n.deadline_by_round = [
                datetime.now() + timedelta(seconds=p)
                for p in np.cumsum(self.inter_ddl_periods_by_ft_id[ft_id])
            ]
            print_dates(n.deadline_by_round, f"DEADLINES FOR CL {self.id}:")
            if self.user_args.partially_available:
                n.set_datasets(n.deadline_by_round)  # node will get data gradually through DatasetPartiallyAvailable
            else:
                n.set_datasets(None)  # node have all the date initially
            # print(f"client {self.id} task {ft_id} data sizes: for train {len(n.local_data.dataset)} for val {len(n.validate_set.dataset)}")
        exit()

    def run(self, read_q, write_q):
        client_start_time = time.time()
        self.setup()
        self.set_deadlines()
        while not self.should_finish:
            self.handle_messages(read_q, write_q)
            ft_id, r = self.scheduler.get_next_task(self.agr_model_by_ft_id_round,
                                                    self.node_by_ft_id, self.user_args.T)
            if ft_id is not None:
                agr_model = self.agr_model_by_ft_id_round[(ft_id, r - 1)]
                ft_args = self.args_by_ft_id[ft_id]
                node = self.node_by_ft_id[ft_id]
                self._set_aggregated_model(ft_args, node, agr_model)
                mean_loss, data_len, start_time, end_time = self._train_one_round(ft_args, node)
                self.data_lens_by_ft_id[ft_id].append(data_len)
                acc = validate(ft_args, node)
                node.iterations_performed += 1  # TODO not to mess with r
                deadline = node.deadline_by_round[r]  # deadline to perform round r
                data_lens = self.data_lens_by_ft_id[ft_id]
                update_quality = data_lens[-1] - data_lens[
                    -2]  # how much new data points was used in this training round
                target_acc = self.args_by_ft_id[ft_id].target_acc
                time_to_target_acc = -1 if (acc < target_acc) else (time.time() - client_start_time)
                # print(f"    Client {self.id} acc is {acc} target_acc is {target_acc} time_to_target_acc is {time_to_target_acc}")
                response = MessageToHub(-1, ft_id,  # TODO delete -1
                                        acc, mean_loss,
                                        copy.deepcopy(node.model),
                                        self.id,
                                        deadline,
                                        update_quality,
                                        r,
                                        Period(start_time, end_time),
                                        time_to_target_acc)
                try:
                    write_q.put(response)
                    self.scheduler.delete_from_plan(ft_id, r)
                except Exception:
                    print(traceback.format_exc())
                # print(f'    Client {self.id} sent local model for round {response.round_num}, task {response.ft_id}')
            # time.sleep(5)

        print(f'    Client {self.id}: CLIENT is DONE')
