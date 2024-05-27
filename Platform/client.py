import argparse
import gc
import os
import time
import traceback
from fl_log import logging_print
from typing import Dict, List, Optional, Tuple

from message import MessageToHub, ControlMessageToClient, ResponseToHub, Period, MessageAgsToClient, \
    PlanMessageToClient, TaskRound
from utils import validate, setup_seed, combine_lists
from utils import *
from server_funct import *
from client_funct import *


class LocalScheduler:
    def __init__(self, trained_ft_id_round):
        self.trained_ft_id_round: set[TaskRound] = trained_ft_id_round

    def mark_as_trained(self, tr):
        """It notifies the scheduler that round is performed successfully, so it should not be scheduled again"""
        self.trained_ft_id_round.add(tr)


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
            return None
        deadline, ft_id = ready[0]
        return TaskRound(ft_id, planning_round_by_ft_id[ft_id])  # task with min deadline


class HubControlledScheduler(LocalScheduler):
    def __init__(self, trained_ft_id_round, user_args, node_by_ft_id):
        super().__init__(trained_ft_id_round)
        rounds = user_args.T
        self.plan: List[TaskRound] = []

    @call_5_sec
    def print_plan(self):
        logging_print(f"CLIENT PLAN: {self.plan}")

    def get_next_task(self, agr_model_by_ft_id_round, node_by_ft_id: Dict[int, Node], rounds_cnt):
        status = {}
        # self.print_plan()
        for tr in self.plan:
            ft_id, r = tr
            if tr in self.trained_ft_id_round:
                raise Exception(f'{tr} in trained: {self.trained_ft_id_round}')
            n: Node = node_by_ft_id[ft_id]
            has_prev_model = (ft_id, r - 1) in agr_model_by_ft_id_round
            data_available = n.data_for_round_is_available(r)
            status[(r, ft_id)] = f"Data {int(data_available)}, prev_model {int(has_prev_model)}"
            if has_prev_model and data_available:
                # logging_print(f"    Client task is chosen {r, ft_id}")
                return tr
        # logging_print(f"    Client plan is {self.plan}. Can not choose task. Status: ")
        # plogging_print(status)
        return None

    def mark_as_trained(self, tr):  # TODO twice
        """It notifies the scheduler that round is performed successfully, so it should not be scheduled again"""
        super().mark_as_trained(tr)
        if tr in self.plan:
            self.plan.remove(tr)


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
                # logging_print(f"    Client task is chosen {r, ft_id}")
                return TaskRound(ft_id, r)
        # logging_print(f"    Client plan is {self.plan}. Can not choose task. Status: ")
        # plogging_print(status)
        return None

    def mark_as_trained(self, tr):
        """It notifies the scheduler that round is performed successfully, so it should not be scheduled again"""
        super().mark_as_trained(tr)
        self.plan.remove(tr)


class Client:
    def __init__(self, id, node_by_ft_id, args_by_ft_id,
                 agr_model_state_by_ft_id_round: Dict[Tuple, ModelTypedState],
                 user_args,
                 inter_ddl_periods_by_ft_id):
        self.id = id
        self.node_by_ft_id = node_by_ft_id
        self.args_by_ft_id = args_by_ft_id
        self.agr_model_by_ft_id_round: Dict[Tuple, ModelTypedState] = agr_model_state_by_ft_id_round
        self.user_args = user_args
        self.should_run = False
        self.data_lens_by_ft_id: Dict[int, List] = {ft_id: [0] for ft_id in node_by_ft_id}
        self.trained_ft_id_round = set()
        self.scheduler = self.get_scheduler(user_args)
        self.inter_ddl_periods_by_ft_id = inter_ddl_periods_by_ft_id
        self.start_time: Optional[datetime] = None

    def get_scheduler(self, user_args):
        if user_args.local_scheduler == "HubControlledScheduler":
            logging_print(f'    Client {self.id} SCHEDULER is set to HubControlledScheduler')
            return HubControlledScheduler(self.trained_ft_id_round, user_args, self.node_by_ft_id)
        elif user_args.local_scheduler == "CyclicalScheduler":
            logging_print(f'    Client {self.id} SCHEDULER is set to CyclicalScheduler')
            return CyclicalScheduler(self.trained_ft_id_round, user_args, self.node_by_ft_id)
        elif user_args.local_scheduler == "MinDeadlineScheduler":
            logging_print(f'    Client {self.id} SCHEDULER is set to MinDeadlineScheduler')
            return MinDeadlineScheduler(self.trained_ft_id_round)
        raise argparse.ArgumentError(user_args.local_scheduler, "Unknown value")

    def client_localTrain(self, args, node, loss=0.0):
        node.model.cuda()
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
            del data
            del target
            del loss_local
        torch.cuda.empty_cache()
        # node.model.cpu()# it will be after validation
        return loss / len(train_loader), len(train_loader) * node.args.batchsize

    def _load_state_to_model(self, ft_args, node, agr_model: ModelTypedState):
        ModelCast.to_model(agr_model, node.model)

    @call_n_sec(3)
    def print_idle(self):
        logging_print(f"    Client {self.id} idle. {datetime.now().isoformat()}")

    def _train_one_round(self, ft_args, node):
        start_time = datetime.now()
        if self.user_args.debug:
            time.sleep(0.2)
            return 0, 0, start_time, datetime.now()
        epoch_losses = []
        data_len = -1
        if ft_args.client_method == 'local_train':
            # logging_print(f'Node {node.num_id} has available batches: {len(node.local_data)}')
            for epoch in range(ft_args.E):
                loss, data_len = self.client_localTrain(ft_args, node)
                epoch_losses.append(loss)
            mean_loss = sum(epoch_losses) / len(epoch_losses)
            end_time = datetime.now()
            gc.collect()
            return mean_loss, data_len, start_time, end_time
        else:
            raise NotImplemented('Still only local_train =(')

    def setup(self):
        setup_seed(self.user_args.random_seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = self.user_args.device
        if not self.user_args.debug:
            torch.cuda.set_device('cuda:' + self.user_args.device)

    @call_5_sec
    def print_hash(self, ags_q):
        logging_print(f"Client hash {self.id} hash {hash(ags_q)}")

    def save_aggregated_model(self, ft_id: int, round_num: int, agr_model_state: ModelTypedState):
        self.agr_model_by_ft_id_round[(ft_id, round_num)] = agr_model_state

    @call_n_sec(3)
    def print_hm(self):
        logging_print(f"Client hm {self.id} {datetime.now().isoformat()}")

    def handle_messages(self, hub_read_q, hub_write_q, ags_q=None):
        while not hub_read_q.empty():
            mes = hub_read_q.get()
            if isinstance(mes, ControlMessageToClient):
                self.start_time = mes.start_time
                self.should_run = mes.should_run
            elif isinstance(mes, TaskRound):
                if not isinstance(self.scheduler, HubControlledScheduler):
                    raise Exception("Scheduler is not HubControlledScheduler")
                self.scheduler.plan.append(mes)
            else:
                raise ValueError(f"Unknown message type {type(mes)}")
            del mes
        if ags_q is None:
            return
        # self.print_hm()
        while not ags_q.empty():
            mes = ags_q.get()
            if isinstance(mes, MessageAgsToClient):
                # logging_print(f"Client {self.id} got MessageAgsToClient {datetime.now().isoformat()}")
                self.save_aggregated_model(mes.ft_id, mes.round_num, mes.agr_model_state)
                required_deadline = self.node_by_ft_id[mes.ft_id].deadline_by_round[mes.round_num]
                delay = max((datetime.now() - required_deadline), timedelta(seconds=0))
                hub_write_q.put(ResponseToHub(self.id, mes.ft_id, mes.round_num, delay))
            elif isinstance(mes, str):
                logging_print(f"Client {self.id} got from AGS: {mes}")
            else:
                raise ValueError(mes)
            del mes

    def set_deadlines(self):
        for ft_id, n in self.node_by_ft_id.items():
            n: Node
            n.deadline_by_round = [
                datetime.now() + timedelta(seconds=p)
                for p in np.cumsum(self.inter_ddl_periods_by_ft_id[ft_id])
            ]
            # print_dates(n.deadline_by_round, f"DEADLINES FOR CL {self.id}:")
            if self.user_args.partially_available:
                n.set_datasets(n.deadline_by_round)  # node will get data gradually through DatasetPartiallyAvailable
            else:
                n.set_datasets(None)  # node have all the data initially
            # logging_print(f"client {self.id} task {ft_id} data sizes: for train {len(n.local_data.dataset)} for val {len(n.validate_set.dataset)}")
        # exit()

    def idle_until_run_cmd(self, read_q, write_q):
        while self.start_time is None:
            self.handle_messages(read_q, write_q)
            # logging_print(f"Client {self.id} waiting run_cmd")
            time.sleep(1)

    def idle_until_start_time(self):
        if datetime.now() < self.start_time:
            delta = (self.start_time - datetime.now()).total_seconds()
            # logging_print(f"client {self.id} WILL WAKE UP in {int(delta)}s")
            time.sleep(delta)

    @call_n_sec(2)
    def print_run(self):
        logging_print(f"Client running. {self.id}")

    def run(self, hub_read_q, write_q, ags_q):
        logging_print(f"Client {self.id} run")
        self.idle_until_run_cmd(hub_read_q, write_q)
        self.idle_until_start_time()
        # logging_print(f"client {self.id} WOKE UP {format_time(datetime.now())}")
        client_start_time = time.time()
        self.setup()
        self.set_deadlines()
        # logging_print("Client really running")
        while self.should_run:
            # self.print_run()
            self.handle_messages(hub_read_q, write_q, ags_q)
            tr: TaskRound = self.scheduler.get_next_task(self.agr_model_by_ft_id_round,
                                                         self.node_by_ft_id, self.user_args.T)
            if tr is None:
                # self.print_idle()
                continue

            ft_id, r = tr
            # logging_print(f"Client {self.id} scheduled task {ft_id} with round {r}")
            agr_model_state = self.agr_model_by_ft_id_round[(ft_id, r - 1)]
            ft_args = self.args_by_ft_id[ft_id]
            node = self.node_by_ft_id[ft_id]
            self._load_state_to_model(ft_args, node, agr_model_state)
            logging_print(f"    Client {self.id} training task {ft_id}, round {r}")
            mean_loss, data_len, start_time, end_time = self._train_one_round(ft_args, node)
            self.data_lens_by_ft_id[ft_id].append(data_len)
            acc = validate(ft_args, node)
            logging_print(f"****Client {self.id} acc is {round(acc, 2)} for task {ft_id}, round {r}")
            node.model.cpu()
            deadline = node.deadline_by_round[r]  # deadline to perform round r
            data_lens = self.data_lens_by_ft_id[ft_id]
            true_update_quality = (data_lens[-1] - data_lens[-2])
            logging_print(f"    Client {self.id} true update_quality is {round(true_update_quality, 2)}")
            update_quality = 1_000  # TODO it is mock. But it's ok for our purposes of studying scheduling.
            # how much new data points was used in this training round
            target_acc = self.args_by_ft_id[ft_id].target_acc
            time_to_target_acc = -1 if (acc < target_acc) else (time.time() - client_start_time)
            # logging_print(f"    Client {self.id} acc is {acc} target_acc is {target_acc} time_to_target_acc is {time_to_target_acc}")
            response = MessageToHub(ft_id=ft_id,
                                    acc=acc, loss=mean_loss,
                                    model_state=ModelCast.to_state(node.model),
                                    client_id=self.id,
                                    deadline=deadline,
                                    update_quality=update_quality,
                                    round_num=r,
                                    period=Period(start_time, end_time),
                                    sample_size=data_len)
            try:
                write_q.put(response)
                # logging_print(f"Client {self.id} sent model for task {ft_id}, round {r}. {datetime.now().isoformat()}")
                self.scheduler.mark_as_trained(tr)
            except Exception:
                logging_print(traceback.format_exc())
            # logging_print(f'    Client {self.id} sent local model for round {response.round_num}, task {response.ft_id}')
        logging_print(f'    Client {self.id}: CLIENT is DONE')
