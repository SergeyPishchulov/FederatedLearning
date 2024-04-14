import collections
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

import os
from datetime import timedelta, datetime
from pprint import pprint
from typing import List, Set, Optional
from utils import timing, norm, normalize_cntr, interpolate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from utils import format_time, ceil_seconds
from message import Period
import seaborn as sns

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


def get_plotly_colors():
    return list(px.colors.qualitative.Plotly)
    return [f'rgb{x}' for x in list(mcolors.BASE_COLORS.values())]


class Statistics:
    def __init__(self, tasks, clients, args):
        self.experiment_name = 'exp'  # f'{args.aggregation_on}|{args.server_method}|partially_available={args.partially_available}'
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
        self.interpolated_jobs_cnt_in_time: Optional[List] = None
        # self.jobs_cnt_in_ags = []
        # self.time_to_target_acc = pd.DataFrame(None, index=[ft.id for ft in tasks],
        #                                        columns=self.client_cols)

        self.time_to_target_acc_by_ft_id = [np.nan] * len(tasks)
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        if not os.path.exists(self.pngs_directory):
            os.makedirs(self.pngs_directory)
        self.start_time: Optional[datetime] = None
        self.fm = None

    def set_start_time(self, start_time):
        self.start_time = start_time

    def get_flood_measure(self):
        """
        Area under plot of jobs cnt by time
        """
        res = 0
        for i, (cur_t, cur_v) in enumerate(self.interpolated_jobs_cnt_in_time):
            if i == 0:
                assert cur_v == 0
                continue
            prev_t, prev_v = self.interpolated_jobs_cnt_in_time[i - 1]
            if cur_v == prev_v:
                res += (cur_t - prev_t).total_seconds() * cur_v
        self.fm = res
        return res

    def print_flood_measure(self):
        fm = self.get_flood_measure()
        print(f"Flood measure: {round(fm, 3)}")

    # @timing
    # def plot_jobs_cnt_in_ags(self):
    #     if not self.jobs_cnt_in_ags:
    #         return
    #     ax = sns.histplot(self.jobs_cnt_in_ags, discrete=True, shrink=0.8, color='royalblue')
    #     plt.xlabel('Number of jobs ready for aggregation')
    #     plt.ylabel('Frequency')
    #     # plt.xticks(list(range(max(data)+1)))
    #     plt.title('Distribution of the number of jobs in AgS')
    #     fig = ax.get_figure()
    #     fig.savefig(f'{self.pngs_directory}/job_cnt_dist.png')

    def set_init_round_beginning(self, task_ids):
        for ft_id in task_ids:
            self.round_done_ts_by_round_num[ft_id] = {-1: datetime.now()}

    @timing
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

    @timing
    def save_ags_period(self, ft_id, period: Period):
        entity = 'agr'
        self.periods_by_entity_ft_id[(entity, ft_id)].append(period)
        periods_by_task = {ft_id: len(periods) for (e, ft_id), periods in self.periods_by_entity_ft_id.items()
                           if e == entity}
        all_periods_cnt = sum([len(periods) for (e, ft_id), periods in self.periods_by_entity_ft_id.items()
                               if e == entity])

    def get_distinguishable_times(self, dt: datetime, uniq_times: Set[datetime]):
        """
        In order not to plot same lines one above another
        """
        while uniq_times and (min(([abs(dt - x) for x in uniq_times])) < timedelta(seconds=0.05)):
            dt = dt + timedelta(seconds=0.05)
        uniq_times.add(dt)
        return dt

    def _plot_jobs_cnt(self, fig):
        if self.interpolated_jobs_cnt_in_time is None:
            return
        interpolated = self.interpolated_jobs_cnt_in_time
        # interpolated = self.jobs_cnt_in_time
        dts, cnts = list(zip(*interpolated))
        fig.add_trace(go.Scatter(
            x=[norm(dt, self.start_time) for dt in dts],
            y=cnts, mode='lines',
            line=dict(color='black',
                      # width=10
                      ),
            legendgroup="jobcnt", name="Job cnt"),
            row=2, col=1)
        # fig.update_xaxes(showticklabels=False, row=2, col=1)

        # fig.update_layout(
        #     yaxis=dict(
        #         tickfont=dict(size=5),
        #         tickmode='array',
        #         # tickvals=list(range(len(y_ticks_texts))),
        #         # ticktext=y_ticks_texts
        #     ),
        #     xaxis=dict(
        #         tickfont=dict(size=20),
        #         title="time",
        #     ),
        #     row=2, col=1 # doesnt work
        # )

    def _plot_first_time_ready_to_aggr(self, first_time_ready_to_aggr, fig, height, colors_by_ft_id,
                                       plotting_period):
        """
        Plot vertical line to show in what moment the decision about
        ability of aggregation was made
        """
        normed_plot_period = None if plotting_period is None else plotting_period.norm(self.start_time)
        if first_time_ready_to_aggr is None:
            return
        uniq_times = set()
        first_time = True
        # pprint({k: format_time(v) for k, v in first_time_ready_to_aggr.items()})
        for (ft_id, r), dt_orig in first_time_ready_to_aggr.items():
            dt = self.get_distinguishable_times(norm(dt_orig, self.start_time), uniq_times)
            if (normed_plot_period is None) or (normed_plot_period.start < dt < normed_plot_period.end):
                fig.add_trace(go.Scatter(
                    x=[dt, dt], y=[0, height - 1], mode='lines',
                    line=dict(color=colors_by_ft_id[ft_id],
                              # width=10
                              ),
                    legendgroup="decision", showlegend=first_time,
                    name="Ready for aggr"),
                    row=1, col=1)
                if first_time:
                    first_time = False
                # axes.plot([dt, dt], [0, height], color=colors_by_ft_id[ft_id])

    def _set_plotly_layout(self, fig, y_ticks_texts):
        # fig.update_layout(font=dict(size=40))
        fig.update_layout(
            yaxis=dict(
                # tickfont=dict(size=40),
                tickmode='array',
                tickvals=list(range(len(y_ticks_texts))),
                ticktext=y_ticks_texts
            ),
            xaxis=dict(
                # tickfont=dict(size=20),
                title="time",
            )
        )
        fm = self.get_flood_measure()
        fig.update_layout(
            title=dict(text=f"System load; fm = {round(fm, 3)}",
                       # font=dict(size=40)
                       )
        )
        return fig

    @timing
    def plot_system_load(self, first_time_ready_to_aggr=None, plotting_period: Period = None):
        """
        Plotting load-plot of clients and AgS

        :param plotting_period: Period of time that will be represented on the plot

        use:     hub.stat.plot_system_load(first_time_ready_to_aggr=hub.journal.first_time_ready_to_aggr)
        """
        ft_ids = sorted(list(set(ft_id for _, ft_id in self.periods_by_entity_ft_id.keys())))
        colors_by_ft_id = get_plotly_colors()[:len(ft_ids)]
        entities = sorted(list(set(e for e, _ in self.periods_by_entity_ft_id.keys())))
        # clients and AgS
        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], shared_xaxes=True)
        for ft_id in ft_ids:
            color = colors_by_ft_id[ft_id]
            legendgroup = f"Task {ft_id}"
            first_scatter_in_group = True
            for i, e in enumerate(entities):
                for (ent, ft_id_2), periods in self.periods_by_entity_ft_id.items():
                    if ent != e or ft_id != ft_id_2:
                        continue
                    for p in periods:
                        p = p.norm(self.start_time)  # TODO check if p in plotting period
                        p: Period
                        fig.add_trace(go.Scatter(
                            x=[p.start, p.end], y=[i] * 2, mode='lines',
                            line=dict(color=color, width=10),
                            legendgroup=legendgroup, showlegend=first_scatter_in_group,
                            name=legendgroup), row=1, col=1)
                        if first_scatter_in_group:
                            first_scatter_in_group = False

        self._set_plotly_layout(fig, entities)
        self._plot_first_time_ready_to_aggr(first_time_ready_to_aggr,
                                            fig,
                                            height=(len(entities)),
                                            colors_by_ft_id=colors_by_ft_id,
                                            plotting_period=plotting_period)
        self._plot_jobs_cnt(fig)
        if plotting_period is None:
            fname = f'{self.pngs_directory}/load'
        else:
            fname = f'{self.pngs_directory}/load_part'
        plotly.offline.plot(fig, filename=fname + ".html")
        fig.write_image(fname + ".png")

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

    # @timing
    def to_csv(self):
        for ft_id, stat_df in self.acc_by_ft_id.items():
            stat_df.to_csv(self.directory + f'/{ft_id}.csv')
        for ft_id, df in self.delay_by_ft_id.items():
            df.to_csv(self.directory + f'/delay_{ft_id}.csv')

    @timing
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
