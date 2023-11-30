from datasets import Data
from nodes import Node
from args import args_parser
from utils import *
from server_funct import *
from client_funct import *
import os


class FederatedMLTaskConfiguration:
    def __init__(self, dataset, nn_architecture, epochs):
        self.dataset = dataset
        self.nn_architecture = nn_architecture
        self.epochs = epochs
        self.iid = 1  # data is iid


class FederatedMLTask:
    def __init__(self, node_cnt, conf: FederatedMLTaskConfiguration,random_seed):
        self.data = Data(conf.dataset, node_cnt, 0, random_seed)  # TODO dumb
        self.conf = conf
        self.node_cnt = node_cnt
        # Data-size-based aggregation weights
        sample_size = []
        for i in range(node_cnt):
            sample_size.append(len(self.data.train_loader[i]))
        self.size_weights = [x / sum(sample_size) for x in sample_size]
        # self.central_node: ClientMLTask = None
        # self.client_nodes: List[ClientMLTask] = None

if __name__ == '__main__':
    args = args_parser()
    setup_seed(args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    torch.cuda.set_device('cuda:' + args.device)

    fedeareted_tasks_configs = []
    fedeareted_tasks_configs.append(FederatedMLTaskConfiguration(
        dataset='cifar10',
        nn_architecture='ResNet20',
        epochs='2'
    ))
    fedeareted_tasks_configs.append(FederatedMLTaskConfiguration(
        dataset='cifar10',
        nn_architecture='ResNet56',
        epochs='1'
    ))
    print(args)
    node_num = 5
    random_seed = 10
    conf = fedeareted_tasks_configs[0]
    fed_task = FederatedMLTask(node_num, conf, random_seed)
    # data = Data(conf.dataset, node_num, 0, random_seed)
    # sample_size = []
    # for i in range(node_num):
        # sample_size.append(len(data.train_loader[i]))
    # size_weights = [i / sum(sample_size) for i in sample_size]

    # Initialize the central node
    # num_id equals to -1 stands for central node
    central_node = Node(-1, fed_task.data.test_loader[0], fed_task.data.test_set, args, node_num)

    # Initialize the client nodes
    client_nodes = {}
    for i in range(fed_task.node_cnt):
        client_nodes[i] = Node(i, fed_task.data.train_loader[i], fed_task.data.train_set, args,node_num)

        # Start the FL training
    final_test_acc_recorder = RunningAverage()
    test_acc_recorder = []
    for rounds in range(args.T):
        print('===============Stage 1 The {:d}-th round==============='.format(rounds + 1))
        lr_scheduler(rounds, client_nodes, args)

        # Client update
        client_nodes, train_loss = Client_update(args, client_nodes, central_node)
        avg_client_acc = Client_validate(args, client_nodes)
        print(args.server_method + args.client_method + ', averaged clients personalization acc is ', avg_client_acc)

        # Partial select function
        if args.select_ratio == 1.0:
            select_list = [idx for idx in range(len(client_nodes))]
        else:
            select_list = generate_selectlist(client_nodes, args.select_ratio)

        # Server update
        central_node = Server_update(args, central_node, client_nodes, select_list, fed_task.size_weights)
        acc = validate(args, central_node, which_dataset='local')
        print(args.server_method + args.client_method + ', global model test acc is ', acc)
        test_acc_recorder.append(acc)

        # Final acc recorder
        if rounds >= args.T - 10:
            final_test_acc_recorder.update(acc)

    print(args.server_method + args.client_method + ', final_testacc is ', final_test_acc_recorder.value())
