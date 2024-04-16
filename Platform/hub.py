import argparse
from datetime import timedelta, datetime
from typing import List, Set

from message import MessageToValidator, Period
from model_cast import ModelTypedState
from aggregation_station import RandomAggregationStationScheduler, SFAggregationStationScheduler
from federated_ml_task import FederatedMLTask
from utils import generate_selectlist, call_5_sec
from training_journal import TrainingJournal
from statistics_handler import Statistics
from torch.multiprocessing import Pool, Process, set_start_method, Queue


class Hub:
    def __init__(self, tasks: List[FederatedMLTask], clients, args, val_write_q):
        self.tasks = tasks
        self.args = args
        self.should_finish = False
        self.val_write_q = val_write_q
        self.clients = clients
        self.stat: Statistics = Statistics(tasks, clients, args)
        self.journal = TrainingJournal([ft.id for ft in tasks], {ft.id: ft.args.required_quality
                                                                 for ft in tasks}, args)
        self.write_q_by_cl_id, self.read_q_by_cl_id = self.init_qs()
        # self._init_scheduler(args)
        # self.finished_by_client = {cl.id: False for cl in clients}
        self.latest_round_with_response_by_ft_id = {t.id: -1 for t in tasks}
        self.sent_jobs_ids: Set = set()
        self.last_plot = datetime.now()
        self._printed = set()
        self.aggregated_jobs = 0
        # self.start_time: datetime.datetime = start_time

    @call_5_sec
    def print_all_done(self):
        print(f"DONE {[ft.done for ft in self.tasks]}")

    @call_5_sec
    def print_progress(self):
        total_aggregations = len(self.tasks) * self.args.T
        print(f"Progress: {self.aggregated_jobs}/{total_aggregations}")

    def all_done(self) -> bool:
        # self.print_all_done()
        return all(ft.done for ft in self.tasks)

    @call_5_sec
    def mark_tasks(self):
        for ft in self.tasks:
            last_round_num = self.args.T - 1
            all_aggregation_done = (ft.latest_agg_round == last_round_num)
            # print(f"FINAL? ft_id={ft.id}, r={ft.latest_agg_round} {self.some_client_got_aggregated_model(ft, last_round_num)}")
            somebody_received = self.some_client_got_aggregated_model(ft, last_round_num)
            # self.debug_print(all_aggregation_done, somebody_received)
            if all_aggregation_done and somebody_received:
                ft.done = True
                mes = f'HUB: Task {ft.id} is done'
            else:
                mes = f'HUB: Performed {ft.latest_agg_round + 1}/{self.args.T} rounds in task {ft.id}'
            if mes not in self._printed:
                print(mes)
                self._printed.add(mes)

    def receive_server_model(self, ft_id):
        return self.tasks[ft_id].central_node

    def some_client_got_aggregated_model(self, ft, round_num):
        return self.latest_round_with_response_by_ft_id[ft.id] == round_num

    @call_5_sec
    def debug_print(self, all_aggregation_done, somebody_received):
        print(f"all_aggregation_done={all_aggregation_done}; somebody_received={somebody_received}")

    def mark_ft_if_done(self, ft_id, ag_round_num):
        ft = self.tasks[ft_id]
        ft.latest_agg_round = max(ft.latest_agg_round, ag_round_num)

    def send_to_validator(self, ft_id, ag_round_num, model_state: ModelTypedState):
        self.val_write_q.put(MessageToValidator(
            ft_id, ag_round_num,
            model_state=model_state
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
