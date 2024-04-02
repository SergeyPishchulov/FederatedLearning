import argparse
from datetime import timedelta, datetime
from typing import List, Set

from message import MessageToValidator, Period
from model_cast import ModelTypedState
from aggregation_station import RandomAggregationStationScheduler, SFAggregationStationScheduler
from federated_ml_task import FederatedMLTask
from utils import generate_selectlist
from training_journal import TrainingJournal
from statistics_handler import Statistics
from torch.multiprocessing import Pool, Process, set_start_method, Queue


class Hub:
    def __init__(self, tasks: List[FederatedMLTask], clients, args, val_write_q):
        self.tasks = tasks
        self.args = args
        self.val_write_q = val_write_q
        self.clients = clients
        self.stat: Statistics = Statistics(tasks, clients, args)
        self.journal = TrainingJournal([ft.id for ft in tasks], {ft.id: ft.args.required_quality
                                                                 for ft in tasks}, args)
        self.write_q_by_cl_id, self.read_q_by_cl_id = self.init_qs()
        # self._init_scheduler(args)
        self.finished_by_client = {cl.id: False for cl in clients}
        self.sent_jobs_ids: Set = set()
        self.last_plot = datetime.now()
        # self.start_time: datetime.datetime = start_time

    # def _init_scheduler(self, args):
    #     if args.aggregation_scheduler == 'random':
    #         self.aggregation_scheduler = RandomAggregationStationScheduler
    #         print(f'RandomAggregationStationScheduler is set')
    #     elif args.aggregation_scheduler == 'SF':
    #         self.aggregation_scheduler = SFAggregationStationScheduler
    #         print(f'SFAggregationStationScheduler is set')
    #     else:
    #         raise argparse.ArgumentError(args.aggregation_scheduler,
    #                                      'Incorrect value for aggregation_scheduler')

    def receive_server_model(self, ft_id):
        return self.tasks[ft_id].central_node

    def mark_ft_if_done(self, ft_id, ag_round_num):
        ft = self.tasks[ft_id]
        all_aggregation_done = (ag_round_num == self.args.T - 1)
        if all_aggregation_done:
            ft.done = True
            print(f'HUB: Task {ft.id} is done')
        else:
            print(f'HUB: Performed {ag_round_num + 1}/{self.args.T} rounds in task {ft.id}')

    def send_to_validator(self, ft_id, ag_round_num, model_state: ModelTypedState):
        self.val_write_q.put(MessageToValidator(
            ft_id, ag_round_num,
            model_state=model_state  # TODO check if ot will work
        ))

    def plot_stat(self):
        if (datetime.now() - self.last_plot) > timedelta(seconds=30):
            self.last_plot = datetime.now()
            self.stat.to_csv()
            self.stat.plot_system_load(first_time_ready_to_aggr=self.journal.first_time_ready_to_aggr)
            # self.stat.plot_jobs_cnt_in_ags()
            # self.stat.print_jobs_cnt_in_ags_statistics()
            self.stat.plot_system_load(plotting_period=Period(self.stat.start_time,
                                                              self.stat.start_time + timedelta(minutes=1)))

    # def get_select_list(self, ft, client_ids):
    #     if ft.args.select_ratio == 1.0:
    #         select_list = client_ids
    #     else:
    #         select_list = generate_selectlist(client_ids, ft.args.select_ratio)
    #     return select_list

    def init_qs(self):
        write_q_by_cl_id = {
            cl.id: Queue()
            for cl in self.clients
        }
        read_q_by_cl_id = {
            cl.id: Queue()
            for cl in self.clients
        }
        return write_q_by_cl_id, read_q_by_cl_id
