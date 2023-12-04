import traceback
from torch.multiprocessing import Pool, Process, set_start_method, Queue
from client import Client
from federated_ml_task import FederatedMLTask
from hub import Hub
import time
from message import MessageToClient, MessageToHub
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

if __name__ == '__main__':
    user_args = args_parser()
    setup_seed(user_args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = user_args.device
    torch.cuda.set_device('cuda:' + user_args.device)

    fedeareted_tasks_configs = get_configs(user_args)
    tasks = [FederatedMLTask(id, c) for id, c in enumerate(fedeareted_tasks_configs)]

    ROUNDS = 10
    clients = []
    print(f'node num is {user_args.node_num}')
    for client_id in range(user_args.node_num):
        clients.append(Client(client_id, {ft.id:
                                              Node(client_id, ft.data.train_loader[client_id],
                                                   ft.data.train_set, ft.args, ft.node_cnt) for ft in tasks},
                              args_by_ft_id={ft.id: ft.args for ft in tasks},
                              agr_model_by_ft_id_round={(ft.id, -1): ft.central_node.model for ft in tasks},
                              user_args=user_args))
    hub = Hub(tasks, clients, user_args)
    final_test_acc_recorder = RunningAverage()
    test_acc_recorder = []

    procs = []
    for client in clients:
        client: Client
        p = Process(target=client.run,
                    args=(hub.write_q_by_cl_id[client.id],
                          hub.read_q_by_cl_id[client.id]))
        procs.append(p)
        p.start()

    # gens = [c.run() for c in clients]
    while not all(ft.done for ft in tasks):
        for cl_id, q in hub.read_q_by_cl_id.items():
            while not q.empty():
                r: MessageToHub = q.get()
                print(f'Got update from client {r.client_id}. Round {r.round_num} for task {r.ft_id} is done')
                hub.journal.save_local(r.ft_id, r.client_id, r.round_num, r.model)
                hub.stat.save_client_ac(r.client_id, r.ft_id, r.round_num, r.acc)

        next_ft_id, ag_round, client_models = hub.journal.get_ft_to_aggregate([c.id for c in clients])
        if next_ft_id is not None:
            ft = tasks[next_ft_id]
            Server_update(ft.args, ft.central_node.model, client_models,
                          hub.get_select_list(ft, [c.id for c in clients]),
                          ft.size_weights)
            hub.journal.mark_as_aggregated(ft.id)
            print(f'AGS Success. Task {ft.id}, round {ag_round}')
            if ag_round == user_args.T:
                tasks[ft.id].done = True
                print(f'Task {ft.id} is done')

            acc = validate(ft.args, ft.central_node, which_dataset='local')
            hub.stat.save_agr_ac(ft.id,
                                 round=ag_round,
                                 acc=acc)
            for c in clients:
                try:
                    hub.write_q_by_cl_id[c.id].put(
                        MessageToClient(ag_round, ft.id,
                                        copy.deepcopy(ft.central_node.model
                                                      )))
                except Exception:
                    print(traceback.format_exc())

                hub.write_q_by_cl_id[c.id].put(
                    MessageToClient(ag_round, ft.id,
                                    copy.deepcopy(ft.central_node.model
                                                  )))

        hub.stat.to_csv()
        hub.stat.plot_accuracy()
        time.sleep(0.5)

    for proc in procs:
        proc.join()
