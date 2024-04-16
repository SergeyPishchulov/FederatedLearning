import argparse
import datetime
import traceback
from datetime import timedelta

from torch.multiprocessing import Process, set_start_method, Queue
from typing import List

from validator import Validator
from ags import AGS
from aggregation_station import Job
from client import Client
from federated_ml_task import FederatedMLTask
from hub import Hub
from message import MessageToHub, ResponseToHub, MessageToValidator, MessageValidatorToHub, \
    ValidatorShouldFinish, ControlMessageToClient, ControlValidatorMessage, MessageHubToAGS, ControlMessageHubToAGS, \
    MessageAgsToHub, FinishMessageToAGS
from config.experiment_config import get_configs
from config.args import args_parser
from utils import *
from server_funct import *
from client_funct import *
import os

try:
    set_start_method('spawn')
except RuntimeError:
    pass


# @timing
def create_clients(tasks, user_args) -> List[Client]:
    clients = []
    inter_ddl_prds = get_interdeadline_periods(tasks, clients_cnt=user_args.node_num)

    for client_id in range(user_args.node_num):
        ft: FederatedMLTask
        node_by_ft_id = {ft.id:
                             Node(client_id, ft.data.train_loader[client_id],
                                  ft.data.train_set, ft.args)
                         for ft in tasks}
        client = Client(client_id,
                        node_by_ft_id,
                        args_by_ft_id={ft.id: ft.args for ft in tasks},
                        agr_model_state_by_ft_id_round={(ft.id, -1): ModelCast.to_state(
                            ft.central_node.model)
                            for ft in tasks},
                        user_args=user_args,
                        inter_ddl_periods_by_ft_id=inter_ddl_prds[client_id]
                        )
        clients.append(client)
    return clients


def get_ags_qs_by_cl_id(clients):
    return {cl.id: Queue() for cl in clients}


def get_client_procs(clients, hub, ags_q_by_cli_id):
    procs = []
    for client in clients:
        client: Client
        p = Process(target=client.run,
                    args=(hub.write_q_by_cl_id[client.id],
                          hub.read_q_by_cl_id[client.id],
                          ags_q_by_cli_id[client.id]))
        procs.append(p)
    return procs


def get_validator_proc(validator: Validator, read_q, write_q):
    return Process(target=validator.run,
                   args=(write_q, read_q))


def get_ags_proc(ags: AGS, read_q, write_q, ags_q_by_cli_id):
    return Process(target=ags.run,
                   args=(write_q, read_q, ags_q_by_cli_id))


# @timing
def handle_messages(hub: Hub, ags_read_q):
    for cl_id, q in hub.read_q_by_cl_id.items():
        while not q.empty():
            r = q.get()
            if isinstance(r, MessageToHub):
                # print(
                #     f'Got update from client {r.client_id}. Round {r.round_num} for task {r.ft_id} is done. DL is {r.deadline}')
                hub.journal.save_local(r.ft_id, r.client_id, r.round_num,
                                       model_state=r.model_state,
                                       deadline=r.deadline,
                                       update_quality=r.update_quality)
                hub.stat.save_client_ac(r.client_id, r.ft_id, r.round_num, r.acc)
                hub.stat.save_client_period(r.client_id, r.ft_id, r.period)
                # hub.stat.print_time_target_acc()
            elif isinstance(r, ResponseToHub):
                # print(f'Received ResponseToHub: {r}')
                hub.latest_round_with_response_by_ft_id[r.ft_id] = max(r.round_num,
                                                                       hub.latest_round_with_response_by_ft_id[r.ft_id])
                # print(hub.latest_round_with_response_by_ft_id)
                hub.stat.save_client_delay(r.client_id, r.ft_id, r.round_num, r.delay)
            elif isinstance(r, MessageValidatorToHub):
                hub.stat.save_agr_ac(r.ft_id,
                                     round_num=r.ag_round_num,
                                     acc=r.acc)
                ft = hub.tasks[r.ft_id]
                hub.stat.save_time_to_target_acc_if_reached(ft, r.acc)
            del r
        while not ags_read_q.empty():
            m = ags_read_q.get()
            if isinstance(m, MessageAgsToHub):
                hub.journal.mark_as_aggregated(ft_id=m.ft_id)
                hub.stat.set_round_done_ts(ft_id=m.ft_id, ag_round_num=m.round_num)
                hub.stat.save_ags_period(m.ft_id, m.period)
                hub.stat.interpolated_jobs_cnt_in_time = interpolate(m.jobs_cnt_in_time)
                hub.mark_ft_if_done(m.ft_id, m.round_num)
                hub.send_to_validator(m.ft_id, m.round_num, m.agr_model_state)
                hub.aggregated_jobs = m.aggregated_jobs
            del m


def finish(hub: Hub, val_write_q):
    print('<<<<<<<<<<<<<<<<All tasks are done>>>>>>>>>>>>>>>>')
    hub.stat.print_delay()
    hub.stat.print_sum_round_duration()
    hub.stat.print_mean_result_acc()
    hub.stat.print_time_target_acc()
    hub.stat.plot_accuracy()
    hub.stat.plot_system_load(first_time_ready_to_aggr=hub.journal.first_time_ready_to_aggr)
    end = datetime.now()
    hub.stat.plot_system_load(first_time_ready_to_aggr=hub.journal.first_time_ready_to_aggr,
                              plotting_period=Period(end - timedelta(minutes=5), end))
    hub.stat.print_flood_measure()
    # hub.stat.plot_jobs_cnt_in_ags()
    # hub.stat.print_jobs_cnt_in_ags_statistics()


def run(tasks: List[FederatedMLTask], hub: Hub,
        clients, user_args, val_read_q, val_write_q, ags_write_q, ags_read_q):
    central_nodes_by_ft_id = {t.id: t.central_node for t in tasks}
    hub.stat.set_init_round_beginning([ft.id for ft in tasks])
    while not hub.should_finish:
        start_time = time.time()
        hub.print_progress()
        handle_messages(hub, ags_read_q)
        ready_jobs_dict = hub.journal.get_ft_to_aggregate(
            [c.id for c in clients], central_nodes_by_ft_id, tasks, hub.sent_jobs_ids)
        if ready_jobs_dict:
            print(f"HUB sent to AGS {len(ready_jobs_dict)} jobs")
            ags_write_q.put(MessageHubToAGS(ready_jobs_dict))
            for j in ready_jobs_dict.values():
                hub.sent_jobs_ids.add(j.id)

        hub.mark_tasks()
        if hub.all_done():
            for cl_id, q in hub.write_q_by_cl_id.items():
                q.put(ControlMessageToClient(should_run=False, start_time=None))
            ags_write_q.put(FinishMessageToAGS())
            val_write_q.put(ValidatorShouldFinish())
            hub.should_finish = True

    hub.plot_stat()
    finish(hub, val_write_q)


INTERDEADLINE_SIGMA = 3


def get_interdeadline_periods(tasks: List[FederatedMLTask], clients_cnt: int):
    res = {}
    for ft in tasks:
        task_mean = ft.args.interdeadline_time_sec  # task's characteristics
        client_means = [task_mean] * clients_cnt  # np.linspace(task_mean - 6, task_mean + 6, clients_cnt)
        for cl_id, cl_m in enumerate(client_means):

            interdeadline_periods = np.random.normal(loc=cl_m, scale=INTERDEADLINE_SIGMA, size=ft.args.T)
            if cl_id not in res:
                res[cl_id] = {}
            res[cl_id][ft.id] = interdeadline_periods
    return res


def wait_while_procs_start(procs):
    while not all(p.is_alive() for p in procs):
        print(f"Hub is waiting while all processes start")
        time.sleep(2)


def set_start_time(client_pipes, val_write_q, ags_write_q, fl_start_time):
    for q in client_pipes:
        q.put(ControlMessageToClient(should_run=True, start_time=fl_start_time))

    ags_write_q.put(ControlMessageHubToAGS(start_time=fl_start_time))
    val_write_q.put(ControlValidatorMessage(fl_start_time))


def main():
    # global_start = time.time()
    user_args = args_parser()
    setup_seed(user_args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = user_args.device
    if not user_args.debug:
        torch.cuda.set_device('cuda:' + user_args.device)

    federated_tasks_configs = get_configs(user_args)
    tasks = [FederatedMLTask(id, c) for id, c in enumerate(federated_tasks_configs)]
    # wakeup_time = datetime.now() + timedelta(seconds=15 * user_args.node_num)
    clients = create_clients(tasks, user_args)

    val_read_q, val_write_q = Queue(), Queue()
    hub = Hub(tasks, clients, user_args, val_write_q)

    ags_read_q, ags_write_q = Queue(), Queue()
    ags_q_by_cl_id = get_ags_qs_by_cl_id(clients)
    central_node_by_ft_id = {t.id: copy.deepcopy(t.central_node) for t in tasks}
    ags = AGS(user_args, central_node_by_ft_id)
    validator = Validator(user_args, node_by_ft_id=copy.deepcopy(central_node_by_ft_id))
    val_proc = get_validator_proc(validator, val_read_q, val_write_q)
    ags_proc = get_ags_proc(ags, ags_read_q, ags_write_q, ags_q_by_cl_id)
    procs = [val_proc, ags_proc] + get_client_procs(clients, hub, ags_q_by_cl_id)
    for p in procs:
        p.start()

    wait_while_procs_start(procs)

    fl_start_time = datetime.now() + timedelta(seconds=5)
    hub.stat.set_start_time(fl_start_time)
    set_start_time(hub.write_q_by_cl_id.values(), val_write_q, ags_write_q, fl_start_time)
    run(tasks, hub, clients, user_args, val_read_q, val_write_q, ags_write_q, ags_read_q)

    for proc in procs:
        proc.join()

    # seconds = round((datetime.now() - wakeup_time).total_seconds())
    # # NOTE we can not trust this time because it includes 1 minute untill all clients will wake up
    # print(f"TOTAL FL TIME IS {seconds} s == {round(seconds / 60., 1)} min")


if __name__ == '__main__':
    main()
