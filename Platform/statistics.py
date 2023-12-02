import os
from typing import List
import pandas as pd
import matplotlib.pyplot as plt

class Statistics:
    def __init__(self, fts, clients, args):
        self.stat_by_ft_id = {
            ft.id: pd.DataFrame(columns=['agr_ac'] + [f'client_{c.id}_ac' for c in clients],
                                index=pd.Series(range(args.T), name='round'))
            for ft in fts}

    def save_client_ac(self, client_id, ft_id, round, acc):
        self.stat_by_ft_id[ft_id].loc[round, f'client_{client_id}_ac'] = acc

    def save_agr_ac(self, ft_id, round, acc):
        self.stat_by_ft_id[ft_id].loc[round, 'agr_ac'] = acc

    def to_csv(self):
        current_directory = os.getcwd()
        directory = os.path.join(current_directory, r'stat')
        if not os.path.exists(directory):
            os.makedirs(directory)
        for ft_id, stat_df in self.stat_by_ft_id.items():
            stat_df.to_csv(directory+f'/{ft_id}.csv')



