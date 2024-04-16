import argparse
import copy
from dataclasses import dataclass, astuple
import time
from model_cast import ModelCast, ModelTypedState
from federated_ml_task import FederatedMLTask
from aggregation_station import Job
import numpy as np
import torch
from datetime import datetime, date

from typing import Dict, List

from utils import timing, print_dates, get_params_cnt

last_print = 0


@dataclass
class JournalRecord:
    """
    Record about message from client

    model - locally trained model
    deadline - max time the aggregation should have been done to
    """
    model_state: ModelTypedState
    deadline: datetime
    update_quality: float


FT_ID = int
ROUND = int


def cpu_copy(jobs: Dict[FT_ID, Job]):
    res = copy.deepcopy(jobs)
    return res


class TrainingJournal:
    def __init__(self, task_ids, required_quality, args):
        self.d: Dict[tuple, JournalRecord] = {}  # key is (ft_id, client_id, round); value is Record(model, deadline)
        self.latest_aggregated_round = {i: -1 for i in task_ids}
        self.required_quality_by_ft_id = required_quality
        self.args = args
        self.first_time_ready_to_aggr: Dict[(FT_ID, ROUND), datetime] = {}

    @timing
    def mark_as_aggregated(self, ft_id):
        self.latest_aggregated_round[ft_id] += 1
        # TODO bug if we skip some rounds

    def save_local(self, ft_id, client_id, round_num, model_state: ModelTypedState, deadline, update_quality):
        if (ft_id, client_id, round_num) not in self.d:
            self.d[(ft_id, client_id, round_num)] = JournalRecord(model_state, deadline, update_quality)
            # print(f"HUB saved {(ft_id, client_id, round_num)}")
        else:
            raise KeyError("Key already exists")

    # @timing
    def get_ft_ready_to_agr(self, client_ids):
        res = []
        if self.args.aggregation_on == 'all_received':
            decision_func = self.all_clients_performed_round
        elif self.args.aggregation_on == 'required_quality':
            decision_func = self.required_quality_reached
        else:
            raise argparse.ArgumentError(self.args.aggregation_on, "Unknown value")
        for ft_id, latest_round in self.latest_aggregated_round.items():
            condition = decision_func(ft_id, latest_round, client_ids)
            if condition:
                res.append((ft_id, latest_round + 1))
                if (ft_id, latest_round + 1) not in self.first_time_ready_to_aggr:
                    self.first_time_ready_to_aggr[(ft_id, latest_round + 1)] = datetime.now()
                    dt = self.first_time_ready_to_aggr[(ft_id, latest_round + 1)]
                    # print_dates([dt], f"Datetime when ready to aggregate. ft_id={ft_id}, latest_round+1={latest_round+1}")
            else:
                global last_print
                if int(time.time()) - last_print > 5:
                    last_print = int(time.time())
                    # print(f"HUB SCHEDULER {self.latest_aggregated_round}")

        return res

    def all_clients_performed_round(self, ft_id, round_num, client_ids):
        return all((ft_id, cl_id, round_num + 1) in self.d
                   for cl_id in client_ids)

    def required_quality_reached(self, ft_id, round_num, client_ids):
        sum_quality = 0
        participants_cnt = 0
        for cl_id in client_ids:
            if (ft_id, cl_id, round_num + 1) in self.d:
                sum_quality += self.d[(ft_id, cl_id, round_num + 1)].update_quality
                participants_cnt += 1
        # print(
        # f"JOURNAL: Quality reached/required = {round(sum_quality / self.required_quality_by_ft_id[ft_id], 3)}. SUM is{sum_quality}")
        if (sum_quality >= 2000  # self.required_quality_by_ft_id[ft_id]
                and participants_cnt >= 2):
            # print(f"JOURNAL: Quality reached.")
            return True
        # print(f"JOURNAL: Quality reached/required = {round(sum_quality / self.required_quality_by_ft_id[ft_id], 3)}")
        return False or self.all_clients_performed_round(ft_id, round_num, client_ids)

    # @timing
    def get_ft_to_aggregate(self, client_ids, central_nodes_by_ft_id, tasks: List[FederatedMLTask],
                            sent_jobs_ids) -> Dict[FT_ID, Job]:
        ready = self.get_ft_ready_to_agr(client_ids)
        if not ready:
            return {}  # TODO optimize: delete already aggregated from ready>
        total_min_deadline = datetime.max
        res_jobs = {}
        for ft_id, round_num in ready:
            # TODO think what will happen dith updates that was sent after aggreagation
            # NOTE: minimal deadline of task is computed only by clients who have sent an update
            # Fair enough: clients who havent sent it can not vote for aggregation of the task
            records = []
            for cl_id in client_ids:
                if (ft_id, cl_id, round_num) in self.d:
                    records.append(self.d[(ft_id, cl_id, round_num)])
            model_states = [r.model_state for r in records]
            min_d = min([r.deadline for r in records])  # feature of the task
            c_node = central_nodes_by_ft_id[ft_id]
            ft = tasks[ft_id]
            job = Job(ft_id, min_d, round_num,
                      processing_time_coef=1,  # MEGA TODO set distinct coefs!!!
                      model_states=model_states,
                      size_weights=ft.size_weights)
            if job.id in sent_jobs_ids:
                continue
            res_jobs[ft_id] = job
            if min_d < total_min_deadline:
                total_min_deadline = min_d
        # print(f'Task {res[1]} with min deadline {res[0]}')
        # return res
        return cpu_copy(res_jobs)

    # print(f"No task to aggregate. d keys: {self.d.keys()}")
