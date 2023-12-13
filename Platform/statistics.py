import copy
import os
from datetime import timedelta, datetime
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class Statistics:
    def __init__(self, tasks, clients, args):
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
        self.directory = os.path.join(current_directory, r'stat')
        self.pngs_directory = os.path.join(current_directory, r'stat/pngs')
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

    def save_client_ac(self, client_id, ft_id, round, acc):
        self.acc_by_ft_id[ft_id].loc[round, f'client_{client_id}'] = acc

    def save_client_delay(self, client_id, ft_id, round, delay):
        self.delay_by_ft_id[ft_id].loc[round, f'client_{client_id}'] = delay

    def save_agr_ac(self, ft_id, round, acc):
        self.acc_by_ft_id[ft_id].loc[round, 'agr'] = acc

    def print_time_target_acc(self, tasks):
        """Prints time required to reach target accuracy for the specified task"""
        res = {}
        for ft in tasks:
            acs = self.acc_by_ft_id[ft.id]['agr']

    def print_delay(self):
        res = timedelta(0)
        for ft_id, df in self.delay_by_ft_id.items():
            # print(df)
            res += df[self.client_cols].sum().sum()
        print(f'SUM_DELAY: {res.total_seconds()} s')

    def plot_delay(self):
        fig, axes = plt.subplots(1, figsize=(10, 8))
        axes.set_title(f"Delay by round. Sum of clients delays for each task")
        for ft_id, df in self.delay_by_ft_id.items():
            sum_by_all_clients = df.map(lambda x: x.total_seconds()).sum(axis=1)
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
            axes[ft_id].set_title(ft_id)
            for c in stat_df.columns:
                if 'client' in c:
                    axes[ft_id].plot(stat_df[c],
                                     # color=color_by_dataset[t.dataset_name],
                                     label=c,
                                     linestyle='dashed')

            axes[ft_id].plot(stat_df['agr'], label='agr_ac')
            axes[ft_id].legend()
        # plt.show()
        fig.savefig(f'{self.pngs_directory}/ac.png')
        plt.close()
