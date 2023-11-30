from typing import List

from federated_ml_task import FederatedMLTaskConfiguration, FederatedMLTask
from fl_args import FLConfig
from args import args_parser
from client_funct import Client_update, Client_validate
from datasets import Data
from nodes import ClientMLTask
from server_funct import receive_client_models, fedlaw_optimization, fedlaw_generate_global_model
from utils import RunningAverage, lr_scheduler, generate_selectlist, validate

from itertools import zip_longest


def combine_lists(*l):
    return [j for i in zip_longest(*l) for j in i if j]


# def make_clients(fed_tasks: List[FederatedMLTask], args):
#     node_cnt = fed_tasks[0].node_cnt
#     clients = [Client(x) for x in range(node_cnt)]
#     task_cnt = len(fed_tasks)
#     for ft in fed_tasks:
#         for i, client in enumerate(clients):
#             client.tasks.append(ClientMLTask(i, ft.data.train_loader[i], ft.data.train_set, args))
#
#     return clients


if __name__ == '__main__':
    args = args_parser()

    fl_config = FLConfig(node_cnt=3, aggr_algorithm='fedavg')

    fedeareted_tasks_configs = []
    fedeareted_tasks_configs.append(FederatedMLTaskConfiguration(
        dataset='cifar10',
        nn_architecture='ResNet20',
        epochs='2'
    ))
    fedeareted_tasks_configs.append(FederatedMLTaskConfiguration(
        dataset='cifar100',
        nn_architecture='ResNet56',
        epochs='1'
    ))

    federated_tasks = []
    for c in fedeareted_tasks_configs:
        ft = FederatedMLTask(fl_config.node_cnt, c)
        ft.central_node = ClientMLTask(-1, ft.data.test_loader[0], ft.data.test_set, args)
        client_nodes = {}
        for i in range(ft.node_cnt):
            client_nodes[i] = ClientMLTask(i, ft.data.train_loader[i], ft.data.train_set, args)
        federated_tasks.append(ft)

    args.server_method = 'fedlaw'
    # Start the FL training
    final_test_acc_recorder = RunningAverage()
    test_acc_recorder = []
    # plan = ([(fedeareted_tasks_configs[0], x) for x in range(10)])  # +
    # [(fedeareted_tasks_configs[1], x) for x in range(10)])
    ft: FederatedMLTask = federated_tasks[0]
    for rounds in range(2):
        print('===============Stage 1 The {:d}-th round==============='.format(rounds + 1))
        lr_scheduler(rounds, ft.client_nodes, args)

        # get aggregated model & train 1 epoch
        ft.client_nodes, train_loss = Client_update(args, ft.client_nodes, ft.central_node)
        avg_client_acc = Client_validate(args, ft.client_nodes)  # validating on its own hold out set
        print('fedlaw, averaged clients acc is ', avg_client_acc)

        # Partial select function
        if args.select_ratio == 1.0:  # sample clients for next round
            select_list = [idx for idx in range(len(ft.client_nodes))]
        else:
            select_list = generate_selectlist(ft.client_nodes, args.select_ratio)

        # FedLAW server update. Aggregation
        agg_weights, client_params = receive_client_models(args, ft.client_nodes, select_list, ft.size_weights)
        gamma, optmized_weights = fedlaw_optimization(args, agg_weights, client_params, ft.central_node)
        ft.central_node = fedlaw_generate_global_model(gamma, optmized_weights, client_params, ft.central_node)
        acc = validate(args, ft.central_node, which_dataset='local')
        # validating aggregated node on data.test_loader[0]
        print('gamma ', gamma)
        print('optmized_weights', optmized_weights)
        print('fedlaw, global model test acc is ', acc)
        test_acc_recorder.append(acc)

        # Final acc recorder
        if rounds >= args.T - 10:
            final_test_acc_recorder.update(acc)

    print(args.server_method + args.client_method + ', final_testacc is ', final_test_acc_recorder.value())
