import os
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


class Statistics:
    def __init__(self, tasks, clients, args):
        self.client_cols = [f'client_{c.id}_ac' for c in clients]
        self.stat_by_ft_id = {
            ft.id: pd.DataFrame(columns=['agr_ac'] + self.client_cols,
                                index=pd.Series(range(args.T), name='round'))
            for ft in tasks}
        current_directory = os.getcwd()
        self.directory = os.path.join(current_directory, r'stat')
        self.pngs_directory = os.path.join(current_directory, r'stat/pngs')
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
        if not os.path.exists(self.pngs_directory):
            os.makedirs(self.pngs_directory)

    def save_client_ac(self, client_id, ft_id, round, acc):
        self.stat_by_ft_id[ft_id].loc[round, f'client_{client_id}_ac'] = acc

    def save_agr_ac(self, ft_id, round, acc):
        self.stat_by_ft_id[ft_id].loc[round, 'agr_ac'] = acc

    def to_csv(self):
        for ft_id, stat_df in self.stat_by_ft_id.items():
            stat_df.to_csv(self.directory + f'/{ft_id}.csv')

    def plot_accuracy(self):
        # colors = list(mcolors.BASE_COLORS.values())
        fig, axes = plt.subplots(len(self.stat_by_ft_id),
                                 figsize=(50, 40*len(self.stat_by_ft_id)))
        for ft_id, stat_df in self.stat_by_ft_id.items():
            axes[ft_id].set_title(ft_id)
            for c in stat_df.columns:
                if 'client' in c:
                    axes[ft_id].plot(stat_df[c],
                              # color=color_by_dataset[t.dataset_name],
                              label=c,
                              linestyle='dashed')

            axes[ft_id].plot(stat_df['agr_ac'], label='agr_ac')
            axes[ft_id].legend()
        plt.show()
        fig.savefig(f'{self.pngs_directory}/ac.png')
