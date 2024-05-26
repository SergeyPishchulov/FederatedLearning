import argparse
from datetime import timedelta, datetime
import random
from typing import List, Set, Optional, Dict

from fl_log import logging_print
from message import MessageToValidator, Period, TaskRound
from model_cast import ModelTypedState
from aggregation_station import RandomAggregationStationScheduler, SFAggregationStationScheduler
from federated_ml_task import FederatedMLTask
from utils import generate_selectlist, call_5_sec, call_n_sec, timing
from training_journal import TrainingJournal
from statistics_handler import Statistics
from torch.multiprocessing import Pool, Process, set_start_method, Queue


@call_n_sec(1)
def print_planning(idle, info=""):
    return
    logging_print(f"Hub planning. {datetime.now().isoformat()}. idle: {idle}, info: {info}")


class ClientModelSelection:
    def __init__(self, cl_ids, rounds_cnt):
        self.scheduled = set()
        self.idle_cl_ids = set(cl_ids)
        self.rounds_cnt = rounds_cnt

    def get_global_plan(self, cl_ids, rounds_cnt, tasks):
        if len(tasks) * 2 != len(cl_ids):
            raise Exception
        global_plan = {}
        for ft_id in tasks:
            for cl_id in [2 * ft_id, 2 * ft_id + 1]:
                global_plan[cl_id] = [TaskRound(ft_id, r) for r in range(rounds_cnt)]
        return global_plan

    def get_cl_plans(self, latest_round_with_response_by_ft_id: Dict):
        res = {}
        print_planning(self.idle_cl_ids, info='before loop')
        for ft_id, trained_round in latest_round_with_response_by_ft_id.items():
            print_planning(self.idle_cl_ids, info='in loop')
            if len(self.idle_cl_ids) < 2:
                break
            new_round = trained_round + 1
            if new_round == self.rounds_cnt:
                continue
            pair = sorted(list(self.idle_cl_ids), key=lambda x: random.random())[:2]
            tr = TaskRound(ft_id, new_round)
            if tr in self.scheduled:
                continue
                # raise Exception(f"LRbyFTID: {latest_round_with_response_by_ft_id},"
                #                 f"tr is {tr},"
                #                 f"scheduled is {self.scheduled}")
            self.scheduled.add(tr)
            for cl in pair:
                res[cl] = tr
                self.idle_cl_ids.remove(cl)
        if res:
            pass
            # logging_print(f"HUB SCHEDULED {res}")
        else:
            # print_empty_scheduled()
            pass
        if all(x == self.rounds_cnt for x in latest_round_with_response_by_ft_id.values()):
            print_nothing_scheduled()
        return res


@call_n_sec(1)
def print_nothing_scheduled():
    print(f"HUB NOTHING TO SCHEDULE {datetime.now().isoformat()}")


@call_n_sec(1)
def print_empty_scheduled():
    print(f"HUB SCHEDULED EMPTY {datetime.now().isoformat()}")


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
        self.selection: Optional[
            ClientModelSelection] = (ClientModelSelection([cl.id for cl in self.clients], self.args.T)
                                     if self.args.local_scheduler == 'HubControlledScheduler'
                                     else None)

    @call_5_sec
    def print_all_done(self):
        logging_print(f"DONE {[ft.done for ft in self.tasks]}")

    @call_5_sec
    def print_progress(self):
        total_aggregations = len(self.tasks) * self.args.T
        logging_print(f"Progress: {self.aggregated_jobs}/{total_aggregations}")

    def all_done(self) -> bool:
        # self.print_all_done()
        return all(ft.done for ft in self.tasks)

    @call_5_sec
    # @timing
    def mark_tasks(self):
        for ft in self.tasks:
            last_round_num = self.args.T - 1
            all_aggregation_done = (ft.latest_agg_round == last_round_num)
            # logging_print(f"FINAL? ft_id={ft.id}, r={ft.latest_agg_round} {self.some_client_got_aggregated_model(ft, last_round_num)}")
            somebody_received = self.some_client_got_aggregated_model(ft, last_round_num)
            if all_aggregation_done and somebody_received:
                ft.done = True
                mes = f'HUB: Task {ft.id} is done'
            else:
                mes = f'HUB: Performed {ft.latest_agg_round + 1}/{self.args.T} rounds in task {ft.id}'
            if mes not in self._printed:
                logging_print(mes)
                self._printed.add(mes)

    def receive_server_model(self, ft_id):
        return self.tasks[ft_id].central_node

    def some_client_got_aggregated_model(self, ft, round_num):
        return self.latest_round_with_response_by_ft_id[ft.id] == round_num

    @call_5_sec
    def debug_prin4t(self, all_aggregation_done, somebody_received):
        logging_print(f"all_aggregation_done={all_aggregation_done}; somebody_received={somebody_received}")

    def mark_ft_if_done(self, ft_id, ag_round_num):
        ft = self.tasks[ft_id]
        ft.latest_agg_round = max(ft.latest_agg_round, ag_round_num)

    def send_to_validator(self, ft_id, ag_round_num, model_state: ModelTypedState):
        self.val_write_q.put(MessageToValidator(
            ft_id, ag_round_num,
            model_state=model_state
        ))

    # @call_n_sec(30)
    def plot_stat(self):
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
