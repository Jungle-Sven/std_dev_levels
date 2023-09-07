import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class Levels:
    def run(self):
        df = self.read_file()        
        df = self.calc_std_dev_levels(df)
        self.plot_levels(df)

    def read_file(self):
        filename = 'data.csv'
        df = pd.read_csv(filename, names=['receive_timestamp', 'price'])
        df['receive_timestamp'] = pd.to_datetime(df['receive_timestamp'])
        df.set_index('receive_timestamp', inplace=True)
        return df

    def calc_std_dev_levels(self, df):
        #log price
        df['price'] = df['price'].apply(np.log)
        #price difference
        df['price_diff'] = df['price'].diff()        
        #std dev
        df['std_dev'] = df['price_diff'].rolling(window=100).std()
        #use rolling mean to find 'zones' of volatility
        df['std_dev_ma'] = df['std_dev'].rolling(3000).mean()
        #thresholds between levels
        std_dev_ma_threshold_1 = df['std_dev_ma'].quantile(0.2)
        std_dev_ma_threshold_2 = df['std_dev_ma'].quantile(0.4)
        std_dev_ma_threshold_3 = df['std_dev_ma'].quantile(0.6)
        std_dev_ma_threshold_4 = df['std_dev_ma'].quantile(0.8)
        # Initialize 'signal' column with zeros
        df['levels'] = 0
        #assign levels based on average std dev  
        df.loc[(df['std_dev_ma'] > std_dev_ma_threshold_1), 'levels'] = 1
        df.loc[(df['std_dev_ma'] > std_dev_ma_threshold_2), 'levels'] = 2
        df.loc[(df['std_dev_ma'] > std_dev_ma_threshold_3), 'levels'] = 3
        df.loc[(df['std_dev_ma'] > std_dev_ma_threshold_4), 'levels'] = 4
        return df
   
    def plot_levels(self, df):
        fig, ax = plt.subplots()
        colors = {
            0:'grey',
            1:'green',
            2: 'blue',
            3: 'red',
            4: 'black'
        }
        scatter = ax.scatter(
            np.reshape(df.index, -1),
            np.reshape(df['price'], -1),
            c=np.reshape(df['levels'].apply(lambda x: colors[x]), -1),
            s=10,
            linewidths=1            
        )
        # Create proxy artists for legend
        legend_labels = [f'Level {level}' for level in colors.keys()]
        legend_handles = [mpatches.Patch(color=colors[level], label=label) for level, label in enumerate(legend_labels)]

        # Create a legend
        ax.legend(handles=legend_handles, title='Levels')

        plt.title('std dev levels')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True)
        plt.show()
            
if __name__ == '__main__':
    v = Levels()
    v.run()
