from typing import Optional, List, Dict

from torch.multiprocessing import Queue
from datetime import datetime
import time

from FederatedLearning.Platform.model_cast import ModelCast
from nodes import Node
from server_funct import Server_update_fedlaw, Server_update
from aggregation_station import RandomAggregationStationScheduler, SFAggregationStationScheduler, Job
from message import MessageHubToAGS, ControlMessageHubToAGS, MessageAgsToClient


def _get_scheduler(args):
    if args.aggregation_scheduler == 'random':
        print(f'RandomAggregationStationScheduler is set')
        return RandomAggregationStationScheduler
    elif args.aggregation_scheduler == 'SF':
        print(f'SFAggregationStationScheduler is set')
        return SFAggregationStationScheduler
    else:
        raise ValueError(f'Incorrect value for aggregation_scheduler: {args.aggregation_scheduler}')


def _get_updater(args):
    if args.server_method == 'fedlaw':
        return Server_update_fedlaw
    if args.server_method == 'fedavg':
        return Server_update
    raise ValueError(f"Unknown server_method: {args.server_method}")


class AGS:
    def __init__(self, user_args, central_node_by_ft_id: Dict[int, Node]):
        self.user_args = user_args
        self.start_time: Optional[datetime] = None
        self.should_finish = False
        self.jobs: List[Job] = []
        self.scheduler = _get_scheduler(user_args)
        self.updater = _get_updater(user_args)
        self.central_node_by_ft_id = central_node_by_ft_id

    def idle_until_run_cmd(self, hub_read_q):
        while self.start_time is None:
            self.handle_messages(hub_read_q)
            # print(f"Client {self.id} waiting run_cmd")
            time.sleep(1)

    def idle_until_start_time(self):
        if datetime.now() < self.start_time:
            delta = (self.start_time - datetime.now()).total_seconds()
            print(f"AGS WILL WAKE UP in {int(delta)}s")
            time.sleep(delta)

    def handle_messages(self, hub_read_q):
        while not hub_read_q.empty():  # 🍒
            mes = hub_read_q.get()
            if isinstance(mes, MessageHubToAGS):
                self.jobs.extend(mes.jobs_by_ft_id.values())
            elif isinstance(mes, ControlMessageHubToAGS):
                self.start_time = mes.start_time
            else:
                raise ValueError(f"Unknown message {mes}")

    def run(self, hub_read_q, hub_write_q, q_by_cl_id: Dict[int, Queue]):
        self.idle_until_run_cmd(hub_read_q)
        self.idle_until_run_cmd(hub_read_q)
        while not self.should_finish:
            self.handle_messages(hub_read_q)
            if self.jobs:
                best_job: Job = self.scheduler.plan_next(self.jobs)
                central_node = self.central_node_by_ft_id[best_job.ft_id]
                period = self.updater(self.user_args,
                                      central_node,
                                      best_job.model_states,
                                      best_job.size_weights)
                self._send_to_clients(central_node.model, q_by_cl_id)

    def _send_to_clients(self, model, q_by_cl_id: Dict[int, Queue]):
        for q in q_by_cl_id.values():
            q.put(MessageAgsToClient(
                ModelCast.to_state(model)
            ))
