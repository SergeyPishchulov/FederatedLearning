import argparse
import datetime
import traceback
from datetime import timedelta

from torch.multiprocessing import Process, set_start_method, Queue
from typing import List

from validator import Validator
from aggregation_station import Job
from client import Client
from federated_ml_task import FederatedMLTask
from hub import Hub
from message import MessageToClient, MessageToHub, ResponseToHub, MessageToValidator, MessageValidatorToHub, \
    ValidatorShouldFinish, ControlMessageToClient, ControlValidatorMessage
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
                        agr_model_by_ft_id_round={(ft.id, -1): ft.central_node.model for ft in tasks},
                        user_args=user_args,
                        inter_ddl_periods_by_ft_id=inter_ddl_prds[client_id]
                        )
        clients.append(client)
    return clients


def get_client_procs(clients, hub):
    procs = []
    for client in clients:
        client: Client
        p = Process(target=client.run,
                    args=(hub.write_q_by_cl_id[client.id],
                          hub.read_q_by_cl_id[client.id]))
        procs.append(p)
    return procs


def get_validator_proc(validator: Validator, read_q, write_q):
    return Process(target=validator.run,
                   args=(write_q, read_q))


# @timing
def handle_messages(hub):
    for cl_id, q in hub.read_q_by_cl_id.items():
        while not q.empty():
            r = q.get()
            if isinstance(r, MessageToHub):
                # print(
                #     f'Got update from client {r.client_id}. Round {r.round_num} for task {r.ft_id} is done. DL is {r.deadline}')
                hub.journal.save_local(r.ft_id, r.client_id, r.round_num, copy.deepcopy(r.model), r.deadline,
                                       r.update_quality)
                hub.stat.save_client_ac(r.client_id, r.ft_id, r.round_num, r.acc, r.time_to_target_acc_sec)
                hub.stat.save_client_period(r.client_id, r.ft_id, r.period)
                # hub.stat.print_time_target_acc()
            elif isinstance(r, ResponseToHub):
                # print(f'Received ResponseToHub: {r}')
                hub.stat.save_client_delay(r.client_id, r.ft_id, r.round_num, r.delay)
                if r.final_message:
                    hub.finished_by_client[r.client_id] = True
            elif isinstance(r, MessageValidatorToHub):
                hub.stat.save_agr_ac(r.ft_id,
                                     round_num=r.ag_round_num,
                                     acc=r.acc)
                ft = hub.tasks[r.ft_id]
                if r.acc > ft.args.target_acc:
                    seconds_spent = int((datetime.datetime.now() - hub.start_time).total_seconds())
                    hub.stat.save_time_to_target_acc(ft.id, seconds_spent)
            del r


@timing
def send_to_validator(val_write_q, ft, ag_round_num):
    val_write_q.put(MessageToValidator(
        ft.id, ag_round_num,
        copy.deepcopy(ft.central_node)  # TODO check if ot will work
    ))


@timing
def send_agr_model_to_clients(clients, hub, ag_round, ft, should_finish: bool):
    for c in clients:
        try:
            hub.write_q_by_cl_id[c.id].put(
                MessageToClient(ag_round, ft.id,
                                copy.deepcopy(ft.central_node.model),
                                should_run=not should_finish  # TODO check if bug
                                ))
        except Exception:
            print(traceback.format_exc())


def get_updater(user_args):
    if user_args.server_method == 'fedlaw':
        return Server_update_fedlaw
    if user_args.server_method == 'fedavg':
        return Server_update
    raise argparse.ArgumentError(user_args.server_method, "Unknown argument value")


def get_params_cnt(model):
    return sum(p.numel() for p in model.parameters())


def finish(hub, val_write_q):
    val_write_q.put(ValidatorShouldFinish())
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
    hub.stat.plot_jobs_cnt_in_ags()
    hub.stat.print_jobs_cnt_in_ags_statistics()


def run(tasks, hub, clients, user_args, val_read_q, val_write_q):
    total_aggregations = 0
    # hub_start_dt = datetime.now()
    hub.stat.set_init_round_beginning([ft.id for ft in tasks])
    updater = get_updater(user_args)
    while not (all(ft.done for ft in tasks) and all(hub.finished_by_client.values())):
        start_time = time.time()
        handle_messages(hub)
        ready_tasks_dict = hub.journal.get_ft_to_aggregate([c.id for c in clients])
        if ready_tasks_dict:
            jobs = [Job(ft_id, deadline, round_num, processing_time_coef=get_params_cnt(models[0]))
                    for ft_id, (deadline, round_num, models) in ready_tasks_dict.items()]
            while jobs:
                best_job = hub.aggregation_scheduler.plan_next(jobs)
                print(f"$$$ {len(jobs)} jobs in AgS")
                hub.stat.upd_jobs_cnt_in_ags(len(jobs))
                next_ft_id = best_job.ft_id
                _, ag_round_num, client_models = ready_tasks_dict[next_ft_id]
                ft = tasks[next_ft_id]
                print(f"=== 1st part {round(time.time() - start_time, 1)}s")
                ag_start_time = time.time()
                print(f"****** ags runs at {datetime.now().isoformat()}")
                p: Period = updater(ft.args, ft.central_node, client_models,
                                    select_list=list(range(len(client_models))),
                                    # NOTE: all ready clients will be aggregated
                                    # hub.get_select_list(ft, [c.id for c in clients]),
                                    size_weights=ft.size_weights)
                print(f"=== 2nd part (AgS works) {round(time.time() - ag_start_time, 1)}s")
                third_part_start_time = time.time()
                # print_dates([p.start,p.end], "Period from updater")
                ft.central_node.model.cpu()
                total_aggregations += 1
                hub.journal.mark_as_aggregated(ft.id)
                hub.stat.set_round_done_ts(ft.id, ag_round_num)
                hub.stat.save_ags_period(ft.id, p)
                hub.stat.plot_system_load(first_time_ready_to_aggr=hub.journal.first_time_ready_to_aggr)
                print(f'AGS Success. Task {ft.id}, round {ag_round_num}. {format_time(datetime.now())}')
                all_aggregation_done = (ag_round_num == user_args.T - 1)
                if all_aggregation_done:
                    ft.done = True
                    print(f'HUB: Task {ft.id} is done')
                else:
                    print(f'HUB: Performed {ag_round_num + 1}/{user_args.T} rounds in task {ft.id}')

                send_to_validator(val_write_q, ft, ag_round_num)
                send_agr_model_to_clients(clients, hub, ag_round_num, ft,
                                          should_finish=all(ft.done for ft in tasks))
                jobs.remove(best_job)
                print(f"=== 3rd part (save the stuff) {round(time.time() - third_part_start_time, 1)}s")
        forth_aprt_start_time = time.time()
        hub.stat.to_csv()
        hub.stat.plot_system_load(first_time_ready_to_aggr=hub.journal.first_time_ready_to_aggr)
        hub.stat.plot_jobs_cnt_in_ags()
        hub.stat.print_jobs_cnt_in_ags_statistics()
        # hub.stat.plot_periods(plotting_period=Period(hub_start_dt, hub_start_dt + timedelta(minutes=1)))
        print(f"=== 4th part (save the stuff if no one is ready) {round(time.time() - forth_aprt_start_time, 1)}s")

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


def set_start_time(client_pipes, val_write_q, fl_start_time):
    for q in client_pipes:
        q.put(ControlMessageToClient(should_run=True, start_time=fl_start_time))

    val_write_q.put(ControlValidatorMessage(fl_start_time))


def main():
    # global_start = time.time()
    user_args = args_parser()
    setup_seed(user_args.random_seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = user_args.device
    torch.cuda.set_device('cuda:' + user_args.device)

    federated_tasks_configs = get_configs(user_args)
    tasks = [FederatedMLTask(id, c) for id, c in enumerate(federated_tasks_configs)]
    # wakeup_time = datetime.now() + timedelta(seconds=15 * user_args.node_num)
    clients = create_clients(tasks, user_args)
    hub = Hub(tasks, clients, user_args)  # TODO check all start time

    val_read_q, val_write_q = Queue(), Queue()
    validator = Validator(user_args)
    val_proc = get_validator_proc(validator, val_read_q, val_write_q)
    procs = [val_proc] + get_client_procs(clients, hub)
    for p in procs:
        p.start()

    print("Hub will wait")
    wait_while_procs_start(procs)

    fl_start_time = datetime.now() + timedelta(seconds=5)
    set_start_time(hub.write_q_by_cl_id.values(), val_write_q, fl_start_time)
    print("Hub waited")
    run(tasks, hub, clients, user_args, val_read_q, val_write_q)

    for proc in procs:
        proc.join()

    # seconds = round((datetime.now() - wakeup_time).total_seconds())
    # # NOTE we can not trust this time because it includes 1 minute untill all clients will wake up
    # print(f"TOTAL FL TIME IS {seconds} s == {round(seconds / 60., 1)} min")


if __name__ == '__main__':
    main()
