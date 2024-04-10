import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--noniid_type', type=str, default='dirichlet',
                        help="iid or dirichlet")
    parser.add_argument('--iid', type=int, default=0,
                        help='set 1 for iid')
    parser.add_argument('--batchsize', type=int, default=128,
                        help="batchsize")
    parser.add_argument('--validate_batchsize', type=int, default=128,
                        help="batchsize")
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5,
                        help="dirichlet_alpha")
    parser.add_argument('--dirichlet_alpha2', type=float, default=False,
                        help="dirichlet_alpha2")
    parser.add_argument('--longtail_proxyset', type=str, default='none',
                        help="longtail_proxyset")
    parser.add_argument('--longtail_clients', type=str, default='none',
                        help="longtail_clients")
    parser.add_argument('--partially_available', type=int, default=0,
                        help="If this flag is raised the clients" +
                             " will get the data gradually by one portion after each deadline")

    # System
    parser.add_argument('--device', type=str, default='0',
                        help="device: {0 = cuda, -1 = cpu}")
    parser.add_argument('--node_num', type=int, default=2,
                        help="Number of nodes")
    parser.add_argument('--T', type=int, default=1,  # 30
                        help="Number of communication rounds")
    parser.add_argument('--E', type=int, default=1,  # 3
                        help="Number of local epochs: E")
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help="Type of algorithms:{mnist, cifar10,cifar100, fmnist}")
    # parser.add_argument('--select_ratio', type=float, default=1.0,
    #                     help="the ratio of client selection in each round")
    parser.add_argument('--local_model', type=str, default='CNN',
                        help='Type of local model: {CNN, ResNet8, AlexNet}')
    parser.add_argument('--random_seed', type=int, default=10,
                        help="random seed for the whole experiment")
    parser.add_argument('--exp_name', type=str, default='FirstTable',
                        help="experiment name")
    parser.add_argument('--aggregation_scheduler', type=str, default='random',
                        help="Scheduler at aggregation station. random or SF")
    parser.add_argument('--aggregation_on', type=str, default='all_received',
                        help="""Event that will run aggregation process. 
                        Options:
                        all_received - All clients have performed the round
                        required_quality - Required quality of updates is reached. You should specify required_quality in an experiment configs""")
    parser.add_argument('--local_scheduler', type=str, default='MinDeadlineScheduler',
                        help="""Strategy for scheduling tasks during local training.
                        Options:
                        CyclicalScheduler. e.g. 3 tasks and 2 rounds. Schedule will be [t1, t2, t3, t1, t2, t3]
                        MinDeadlineScheduler
                        """)

    # Server function
    parser.add_argument('--server_method', type=str, default='fedavg',
                        help="fedavg, feddf, fedbe, finetune, feddyn, fedadam, fedlaw")
    parser.add_argument('--server_epochs', type=int, default=20,
                        help="optimizer epochs on server, change it to 1, 2, 3, 5, 10")
    parser.add_argument('--server_optimizer', type=str, default='adam',
                        help="type of server optimizer for FedLAW, FedDF, FedBE, finetune, adam or sgd")
    parser.add_argument('--server_valid_ratio', type=float, default=0.02,
                        help="the ratio of validate set (proxy dataset) in the central server")
    parser.add_argument('--server_funct', type=str, default='exp',
                        help="server funct for FedLAW, exp or quad")
    parser.add_argument('--whether_swa', type=str, default='none',
                        help='none or swa for FedLAW')
    parser.add_argument('--fedadam_server_lr', type=float, default=1.0,
                        help="server_lr for FedAdam")

    # Client function
    parser.add_argument('--client_method', type=str, default='local_train',
                        help="local_train, fedprox, feddyn")
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help="optimizer: {sgd, adam}")
    parser.add_argument('--client_valid_ratio', type=float, default=0.3,
                        help="the ratio of validate set in the clients")
    parser.add_argument('--lr', type=float, default=0.08,
                        help='clients loca learning rate')
    parser.add_argument('--local_wd_rate', type=float, default=5e-4,
                        help='clients local wd rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='clients SGD momentum')
    parser.add_argument('--mu', type=float, default=0.001,
                        help="clients proximal term mu for FedProx")
    parser.add_argument('--debug', type=bool, default=False,
                        help="Debug")

    args = parser.parse_args()

    return args
