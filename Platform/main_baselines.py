import traceback
from datetime import datetime, timedelta

from torch.multiprocessing import Pool, Process, set_start_method, Queue
from client import Client
from federated_ml_task import FederatedMLTask
from hub import Hub
import time
from message import MessageToClient, MessageToHub, ResponseToHub
from experiment_config import get_configs
from args import args_parser
from utils import *
from server_funct import *
from client_funct import *
import os

try:
    set_start_method('spawn')
except RuntimeError:
    pass


def create_clients(tasks, user_args):
    clients = []
    for client_id in range(user_args.node_num):
        node_by_ft_id = {ft.id:
                             Node(client_id, ft.data.train_loader[client_id],
                                  ft.data.train_set, ft.args)
                         for ft in tasks}
        client = Client(client_id,
                        node_by_ft_id,
                        args_by_ft_id={ft.id: ft.args for ft in tasks},
                        agr_model_by_ft_id_round={(ft.id, -1): ft.central_node.model for ft in tasks},
                        user_args=user_args)
        clients.append(client)
    return clients


def get_client_procs(clients, hub):
    procs = []
    for client in clients:
        p = Process(target=client.run,
                    args=(hub.write_q_by_cl_id[client.id],
                          hub.read_q_by_cl_id[client.id]))
        procs.append(p)
    return procs


def handle_messages(hub):
    for cl_id, q in hub.read_q_by_cl_id.items():
        while not q.empty():
            r = q.get()
            if isinstance(r, MessageToHub):
                # TODO understand what round is performed
                print(
                    f'Got update from client {r.client_id}. Round {r.iteration_num} for task {r.ft_id} is done. DL is {r.deadline}')
                hub.journal.save_local(r.ft_id, r.client_id, r.iteration_num, copy.deepcopy(r.model), r.deadline)
                hub.stat.save_client_ac(r.client_id, r.ft_id, r.iteration_num, r.acc)
            elif isinstance(r, ResponseToHub):
                # print(f'Received ResponseToHub: {r}')
                hub.stat.save_client_delay(r.client_id, r.ft_id, r.round_num, r.delay)
            del r


def send_agr_model_to_clients(clients, hub, ag_round, ft, should_finish):
    for c in clients:
        try:
            hub.write_q_by_cl_id[c.id].put(
                MessageToClient(ag_round, ft.id,
                                copy.deepcopy(ft.central_node.model),
                                should_finish=should_finish
                                ))
        except Exception:
            print(traceback.format_exc())


def run(tasks, hub, clients, user_args):
    hub.stat.set_init_round_beginning()
    while not all(ft.done for ft in tasks):
        handle_messages(hub)
        _, next_ft_id, ag_round_num, client_models = hub.journal.get_ft_to_aggregate([c.id for c in clients])
        if next_ft_id is not None:
            ft = tasks[next_ft_id]
            Server_update(ft.args, ft.central_node.model, client_models,
                          hub.get_select_list(ft, [c.id for c in clients]),
                          ft.size_weights)
            hub.journal.mark_as_aggregated(ft.id)
            hub.stat.set_round_done_ts(ft.id, ag_round_num)
            print(f'AGS Success. Task {ft.id}, round {ag_round_num}')
            all_aggregation_done = (ag_round_num == user_args.T - 1)
            if all_aggregation_done:
                ft.done = True
                print(f'Task {ft.id} is done')
            else:
                print(f'Performed {ag_round_num + 1}/{user_args.T} rounds in task {ft.id}')

            acc = validate(ft.args, ft.central_node, which_dataset='local')
            hub.stat.save_agr_ac(ft.id,
                                 round=ag_round_num,
                                 acc=acc)
            send_agr_model_to_clients(clients, hub, ag_round_num, ft,
                                      should_finish=all(ft.done for ft in tasks))

        hub.stat.to_csv()
        hub.stat.plot_accuracy()
        hub.stat.plot_delay()
        # time.sleep(0.5)
    print('All tasks are done')
    hub.stat.print_delay()
    hub.stat.print_sum_round_duration()


def main():
    user_args = args_parser()
    setup_seed(user_args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = user_args.device
    torch.cuda.set_device('cuda:' + user_args.device)

    federated_tasks_configs = get_configs(user_args)
    tasks = [FederatedMLTask(id, c) for id, c in enumerate(federated_tasks_configs)]
    clients = create_clients(tasks, user_args)
    hub = Hub(tasks, clients, user_args)
    procs = get_client_procs(clients, hub)
    for p in procs:
        p.start()

    run(tasks, hub, clients, user_args)

    for proc in procs:
        proc.join()


if __name__ == '__main__':
    main()
