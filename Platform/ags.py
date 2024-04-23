from typing import Optional, List, Dict, Set, Tuple

from torch.multiprocessing import Queue
from datetime import datetime
import time

from utils import timing
from model_cast import ModelCast
from nodes import Node
from server_funct import Server_update_fedlaw, Server_update, debug_server_update
from aggregation_station import RandomAggregationStationScheduler, SFAggregationStationScheduler, Job
from message import MessageHubToAGS, ControlMessageHubToAGS, MessageAgsToClient, MessageAgsToHub, Period, \
    FinishMessageToAGS


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
    if args.debug:
        return debug_server_update
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
        self.jobs: Set[Job] = set()
        self.scheduler = _get_scheduler(user_args)
        self.updater = _get_updater(user_args)
        self.central_node_by_ft_id = central_node_by_ft_id
        self.aggregated_jobs = 0
        self.jobs_cnt_in_time: List[Tuple[datetime, int]] = []

    def idle_until_run_cmd(self, hub_read_q):
        while self.start_time is None:
            self.handle_messages(hub_read_q)
            # print(f"Client {self.id} waiting run_cmd")
            time.sleep(1)

    def idle_until_start_time(self):
        if datetime.now() < self.start_time:
            # self.jobs_cnt_in_time.append((self.start_time, 0))
            delta = (self.start_time - datetime.now()).total_seconds()
            print(f"AGS WILL WAKE UP in {int(delta)}s")
            time.sleep(delta)

    def handle_messages(self, hub_read_q):
        while not hub_read_q.empty():  # 🍒
            mes = hub_read_q.get()
            if isinstance(mes, MessageHubToAGS):
                print(f"AGS got {len(mes.jobs_by_ft_id)} jobs")
                self.jobs.update(mes.jobs_by_ft_id.values())
            elif isinstance(mes, ControlMessageHubToAGS):
                self.start_time = mes.start_time
            elif isinstance(mes, FinishMessageToAGS):
                self.should_finish = True
            else:
                raise ValueError(f"Unknown message {mes}")

    def register_jobs_cnt(self):
        new_cnt = len(self.jobs)
        new_record = (datetime.now(), new_cnt)
        if self.jobs_cnt_in_time:
            last_record = self.jobs_cnt_in_time[-1]
            t, cnt = last_record
            if new_cnt == cnt:
                print(f"DBG RJC {datetime.now().isoformat()} new_cnt={new_cnt}, replacing {self.jobs_cnt_in_time[-1]}")
                self.jobs_cnt_in_time[-1] = new_record
                return
        self.jobs_cnt_in_time.append(new_record)

    def run(self, hub_read_q, hub_write_q, q_by_cl_id: Dict[int, Queue]):
        self.idle_until_run_cmd(hub_read_q)
        self.idle_until_start_time()
        print(f"AGS woke up")
        while not self.should_finish:
            self.register_jobs_cnt()
            self.handle_messages(hub_read_q)
            if self.jobs:
                # print(f"AGS scheduling jobs")
                best_job: Job = self.scheduler.plan_next(self.jobs)
                central_node = self.central_node_by_ft_id[best_job.ft_id]
                self.register_jobs_cnt()
                print(f"DBG before update {datetime.now().isoformat()}, jc={len(self.jobs)}, repo={self.jobs_cnt_in_time}")
                period = self.updater(self.user_args,
                                      central_node,
                                      best_job.model_states,
                                      best_job.size_weights)
                print(f"DBG after update {datetime.now().isoformat()}, jc={len(self.jobs)}, repo={self.jobs_cnt_in_time}")
                print(f"AGS Success for task {best_job.ft_id} round {best_job.round_num}")
                self.jobs.remove(best_job)
                print(f"DBG after removing {datetime.now().isoformat()}, jc={len(self.jobs)}, repo={self.jobs_cnt_in_time}")
                self.aggregated_jobs += 1
                self.register_jobs_cnt()
                print(f"DBG after register {datetime.now().isoformat()}, jc={len(self.jobs)}, repo={self.jobs_cnt_in_time}")
                self._send_to_clients(ft_id=best_job.ft_id, round_num=best_job.round_num,
                                      model=central_node.model, q_by_cl_id=q_by_cl_id)
                self.register_jobs_cnt()
                self._notify_hub(ft_id=best_job.ft_id,
                                 round_num=best_job.round_num,
                                 hub_write_q=hub_write_q,
                                 period=period,
                                 model=central_node.model)
            # self.should_finish = True
        self.finish()

    def _send_to_clients(self, ft_id, round_num, model, q_by_cl_id: Dict[int, Queue]):
        for cl_id, q in q_by_cl_id.items():
            q.put(MessageAgsToClient(
                ft_id=ft_id,
                round_num=round_num,
                agr_model_state=ModelCast.to_state(model)
            ))
        # print(f"AGS sent model to clients")

    # @timing
    def _notify_hub(self, ft_id, round_num, hub_write_q, period: Period, model):
        hub_write_q.put(MessageAgsToHub(ft_id=ft_id,
                                        round_num=round_num,
                                        agr_model_state=ModelCast.to_state(model),
                                        period=period,
                                        jobs_cnt_in_time=self.jobs_cnt_in_time,
                                        aggregated_jobs=self.aggregated_jobs))

    def finish(self):
        print(f"AGS finished")
