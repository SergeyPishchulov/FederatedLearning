import copy
import itertools
import os
from datetime import timedelta, datetime
from pprint import pprint
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from message import Period


class Statistics:
    def __init__(self, tasks, clients, args):
        self.experiment_name = f'{args.aggregation_on}|{args.server_method}|partially_available={args.partially_available}'
        self.client_cols = [f'client_{c.id}' for c in clients]
        self.acc_by_ft_id = {
            ft.id: pd.DataFrame(columns=['agr'] + self.client_cols,
                                index=pd.Series(range(args.T), name='round'))
            for ft in tasks}
        self.delay_by_ft_id = {
            ft.id: pd.DataFrame(timedelta(0), columns=self.client_cols,
                                index=pd.Series(range(args.T), name='round'))
            for ft in tasks}
        self.round_done_ts_by_round_num = {}
        current_directory = os.getcwd()
        self.directory = os.path.join(current_directory, r'stat/' + f"{self.experiment_name}")
        self.pngs_directory = os.path.join(self.directory, 'pngs')  # os.path.join(current_directory, r'stat/pngs')
        entities = [f'client_{cl.id}' for cl in clients] + ['agr']
        self.periods_by_entity_ft_id = {(e, t.id): [] for e in entities for t in tasks}
        # self.time_to_target_acc = pd.DataFrame(None, index=[ft.id for ft in tasks],
        #                                        columns=self.client_cols)

        self.time_to_target_acc_by_ft_id = [np.nan] * len(tasks)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        if not os.path.exists(self.pngs_directory):
            os.makedirs(self.pngs_directory)

    def set_init_round_beginning(self, task_ids):
        for ft_id in task_ids:
            self.round_done_ts_by_round_num[ft_id] = {-1: datetime.now()}

    def set_round_done_ts(self, ft_id, ag_round_num):
        """Save moment in time at which ag_round_num is done"""
        self.round_done_ts_by_round_num[ft_id][ag_round_num] = datetime.now()

    def print_sum_round_duration(self):
        res = timedelta(0)
        for ft_id, ts_by_round in self.round_done_ts_by_round_num.items():
            tss = (list(ts_by_round.values()))
            all_rounds_duration = max(tss) - min(tss)
            res += all_rounds_duration
        print(f"All rounds duration (sum by all tasks) {res.total_seconds()} s")

    def save_client_ac(self, client_id, ft_id, round_num, acc, time_to_target_acc_sec):
        self.acc_by_ft_id[ft_id].loc[round_num, f'client_{client_id}'] = acc
        # print(f"Saved acc for {f'client_{client_id}'} is {round(acc, 3)}")
        # if (time_to_target_acc_sec != -1):
        # print("Target acc is reached")
        # if (pd.isnull(self.time_to_target_acc.loc[ft_id, f'client_{client_id}'])):
        #     self.time_to_target_acc.loc[ft_id, f'client_{client_id}'] = time_to_target_acc_sec
        # print("SAVED")
        # print(self.time_to_target_acc)

    def save_client_period(self, client_id, ft_id, period: Period):
        entity = f'client_{client_id}'
        self.periods_by_entity_ft_id[(entity, ft_id)].append(period)

    def save_ags_period(self, ft_id, period: Period):
        entity = 'agr'
        self.periods_by_entity_ft_id[(entity, ft_id)].append(period)
        periods_by_task = {ft_id: len(periods) for (e, ft_id), periods in self.periods_by_entity_ft_id.items()
                           if e == entity}
        all_periods_cnt = sum([len(periods) for (e, ft_id), periods in self.periods_by_entity_ft_id.items()
                               if e == entity])

    def plot_periods(self):
        fig, axes = plt.subplots(1,
                                 figsize=(16, 5)
                                 )

        ft_ids = sorted(list(set(ft_id for _, ft_id in self.periods_by_entity_ft_id.keys())))
        colors_by_ft_id = list(mcolors.BASE_COLORS.values())[:len(ft_ids)]
        entities = sorted(list(set(e for e, _ in self.periods_by_entity_ft_id.keys())))
        # print(f'Entities {entities}, e_ft_id{self.periods_by_entity_ft_id.keys()}')
        for i, e in enumerate(entities):
            total_aggragations = 0
            agr_periods = []
            for (ent, ft_id), periods in self.periods_by_entity_ft_id.items():
                if ent != e:
                    continue
                    # print(f'Plot for ent {e} task {ft_id}')
                for p in periods:
                    if ent == 'agr':
                        total_aggragations += 1
                        agr_periods.append(p)
                    p: Period
                    axes.plot([p.start, p.end], [i] * 2, color=colors_by_ft_id[ft_id],
                              linewidth=10
                              )

        plt.yticks(list(range(len(entities))))
        axes.set_yticklabels(entities, fontsize=20)
        plt.xlabel('time', fontsize=20)
        # plt.legend([f"Task {ft_id}" for ft_id in ft_ids])#BUG
        fig.savefig(f'{self.pngs_directory}/periods.png')
        plt.close()

    def save_client_delay(self, client_id, ft_id, round, delay):
        self.delay_by_ft_id[ft_id].loc[round, f'client_{client_id}'] = delay

    def save_agr_ac(self, ft_id, round_num, acc):
        self.acc_by_ft_id[ft_id].loc[round_num, 'agr'] = acc

    def save_time_to_target_acc(self, ft_id, t):
        if self.time_to_target_acc_by_ft_id[ft_id] is np.nan:
            self.time_to_target_acc_by_ft_id[ft_id] = t

    def print_mean_result_acc(self):
        mean_accs = [df.mean(axis=1).iloc[-1] for df in self.acc_by_ft_id.values()]
        res = round(np.mean(mean_accs))
        print(f"MEAN RESULT ACCURACY IS {res}")

    def print_time_target_acc(self):
        """Prints time required to reach target accuracy for the all tasks"""
        metric_value = pd.Series(self.time_to_target_acc_by_ft_id).mean(skipna=False).round()
        print(f"TIME TO TARGET ACC = {metric_value}")

    def print_delay(self):
        res = timedelta(0)
        for ft_id, df in self.delay_by_ft_id.items():
            # print(df)
            name = f"{self.pngs_directory}/{self.experiment_name}" + f'|ft_id{ft_id}.csv'
            df[self.client_cols].to_csv(name)
            res += df[self.client_cols].sum().sum()
        print(f'SUM_DELAY: {round(res.total_seconds())} s')

    def plot_delay(self):
        fig, axes = plt.subplots(1, figsize=(10, 8))
        axes.set_title(f"Delay by round. Sum of clients delays for each task")
        for ft_id, df in self.delay_by_ft_id.items():
            sum_by_all_clients = np.vectorize(lambda x: x.total_seconds())(df).sum(axis=1)  # bug
            # sum_by_all_clients = df.map(lambda x: x.total_seconds()).sum(axis=1)
            axes.plot(sum_by_all_clients,
                      label=f'task {ft_id}',
                      # linestyle='dashed'
                      )
        axes.set_xlabel('round')
        axes.set_ylabel('Delay (s)')
        axes.legend()
        fig.savefig(f'{self.pngs_directory}/delay.png')
        plt.close()

    def to_csv(self):
        for ft_id, stat_df in self.acc_by_ft_id.items():
            stat_df.to_csv(self.directory + f'/{ft_id}.csv')
        for ft_id, df in self.delay_by_ft_id.items():
            df.to_csv(self.directory + f'/delay_{ft_id}.csv')

    def plot_accuracy(self):
        # colors = list(mcolors.BASE_COLORS.values())
        fig, axes = plt.subplots(len(self.acc_by_ft_id),
                                 figsize=(10, 8 * len(self.acc_by_ft_id)))
        if len(self.acc_by_ft_id) == 1:
            axes = [axes]
        for ft_id, stat_df in self.acc_by_ft_id.items():
            axes[ft_id].set_title(f"Accuracy for task {ft_id}", fontsize=20)
            for c in stat_df.columns:
                if 'client' in c:
                    axes[ft_id].plot(stat_df[c],
                                     # color=color_by_dataset[t.dataset_name],
                                     label=c,
                                     linestyle='dashed')

            axes[ft_id].plot(stat_df['agr'], label='agr_ac')
            axes[ft_id].legend()
            axes[ft_id].tick_params(axis='both', which='major', labelsize=20)
            axes[ft_id].set_xlabel('№ round', fontsize=20)
            axes[ft_id].set_ylabel('Accuracy', fontsize=20)
            # plt.xlabel('№ round', fontsize=20)
            # plt.ylabel('Accuracy', fontsize=20)
        fig.savefig(f'{self.pngs_directory}/ac.png')
        plt.close()
