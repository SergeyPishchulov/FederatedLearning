from typing import List

from federated_ml_task import FederatedMLTask
from utils import generate_selectlist
from training_journal import TrainingJournal
from statistics import Statistics
from torch.multiprocessing import Pool, Process, set_start_method, Queue


class Hub:
    def __init__(self, tasks: List[FederatedMLTask], clients, args):
        self.tasks = tasks
        self.clients = clients
        self.stat = Statistics(tasks, clients, args)
        self.journal = TrainingJournal([ft.id for ft in tasks])
        self.write_q_by_cl_id, self.read_q_by_cl_id = self.init_qs()

    def receive_server_model(self, ft_id):
        return self.tasks[ft_id].central_node

    def get_select_list(self, ft, client_ids):
        if ft.args.select_ratio == 1.0:
            select_list = client_ids
        else:
            select_list = generate_selectlist(client_ids, ft.args.select_ratio)
        return select_list

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
