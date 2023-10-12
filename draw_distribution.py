import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict


def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    return x, y

def update_df(df, dic, save_path):
    for key in set(dic.keys()) - set(df.columns):
        df[key] = None
    df.loc[len(df)] = dic
    df.to_csv(save_path, index=False)

def calculate_quantile(data, percentile):
    sorted_data = np.sort(data)
    index = (percentile / 100) * (len(data) - 1) + 1

    if index.is_integer():
        quantile = sorted_data[int(index)]
    else:
        lower_index = int(np.floor(index))
        upper_index = int(np.ceil(index))
        lower_value = sorted_data[lower_index]
        upper_value = sorted_data[upper_index]
        quantile = lower_value + (index - lower_index) * (upper_value - lower_value)

    return quantile

def plot_d(x, name="draw_d.png", x_name='xi', y_name='cnt', dirname='ConvE-assumption1-inf', typ='t'):
    # plot the distribution of x, draw a histogram and a distribution curve in the same figure
    plt.figure(figsize=(10, 6))
    plt.hist(x, bins=100, density=True, color='blue', alpha=0.5)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.grid()
    # plt.savefig(f'{dirname}/{name}')
    print('len(X):', len(x))

    # fit a distribution curve for x
    from scipy.stats import t

    # Fit the data to a Student's t-distribution
    degree, loc, scale = t.fit(x)
    print(f'degree: {degree}, location: {loc}, scale: {scale}')
    xmin, xmax = plt.xlim()
    x_axis = np.linspace(xmin, xmax, 100)
    p = t.pdf(x_axis, degree, loc, scale)
    plt.plot(x_axis, p, 'k', linewidth=2)
    # add a text to show the parameters of the fitted distribution, on the top right corner
    plt.text(xmax * 0.6, p.max() * 0.9, f'degrees: {degree:.3f}\nlocation: {loc:.3f}\nscale: {scale:.3f}', fontsize=12)
    plt.savefig(f'{dirname}/{name}')


    x_ecdf, y_ecdf = ecdf(x)
    # Calculate CDF
    x_cdf = np.sort(x)
    y_cdf = t.cdf(x_cdf, df=degree, loc=loc, scale=scale)

    plt.figure(figsize=(10, 6))
    plt.plot(x_ecdf, y_ecdf, label='ECDF')
    plt.plot(x_cdf, y_cdf, label='CDF')
    plt.xlabel('Value')
    plt.ylabel('Probability')
    plt.legend()
    plt.savefig(f'{dirname}/ecdf_{name}')


    # do a ks test and a anderson test for x
    from scipy.stats import kstest, anderson, shapiro

    partition_size = 50
    num_partitions = len(x) // partition_size
    partitions = []
    indexes = np.random.choice(len(x), size=num_partitions * partition_size, replace=False)

    for i in range(num_partitions):
        ids = indexes[i * partition_size: (i + 1) * partition_size]
        partition = x[ids]
        partitions.append(partition)
    
    # Do a KS test
    ks_statistics = []
    p_values = []
    for partition in partitions:
        ks_statistic, p_value = kstest(partition, 't', args=(degree, loc, scale))
        ks_statistics.append(ks_statistic)
        p_values.append(p_value)
    ks_statistic, p_value = round(np.mean(ks_statistics), 4), round(np.mean(p_values), 4)
    print(f'KS test statistic: {ks_statistic}, p-value: {p_value}')

    return {'perspective': x_name , 'KS': ks_statistic, 'p_value': p_value, 'degree': round(degree, 4), 'loc': round(loc, 4), 'scale': round(scale, 4)}


def plot(x, y, name='draw_assumption.png', x_name='partial * delta_h * delta_t', y_name='rel - head_rel - tail_rel', dirname='ConvE-assumption2-inf'):
    plt.figure(figsize=(6, 10))
    plt.scatter(x, y, s=5)
    # make sure the x axis and y_axis have the same scale
    max_val = max(max(x), max(abs(y)))
    plt.xlim(0, max_val)
    plt.ylim(-max_val, max_val)

    # make sure length of x-axis and y-axis are the same
    plt.gca().set_aspect('equal', adjustable='box')
    x = np.linspace(0, max_val, 100)

    plt.plot(x, x, color='red')
    plt.plot(x, -x, color='red')

    plt.ylabel(y_name)
    plt.xlabel(x_name)
    plt.grid()
    plt.savefig(f'{dirname}/{name}')


def plot_k(x, y, name="draw_k.png", x_name='k', y_name='ratio', dirname='ConvE-assumption2-inf'):
    ratio = []
    k_range = np.arange(0.001, 1, 0.001)
    # select k from 0 to 2, step=0.1. 

    # calculate how many points are in the red area
    # the red area is the area between the two red lines
    valid = None
    valid_ratio = None
    for k in k_range:
        satisfy = len(y[(y > -x * k) & (y < x * k)])
        # print(satisfy, len(y), satisfy / len(y))
        r = satisfy / len(y)
        ratio.append(r)
        if r > 0.95 and valid is None:
            valid = k
            valid_ratio = r

    if valid is None:
        valid = 0
        valid_ratio = 0

    print(f'valid k: {valid}, valid ratio: {valid_ratio}')
    plt.figure(figsize=(10, 10))
    # draw the point (valid, valid_ratio) in red color 
    plt.plot(k_range, ratio)
    plt.scatter(valid, valid_ratio, s=30, c='red')  
    # add a text to show the valid k and valid ratio
    plt.text(valid, valid_ratio, f'({round(valid, 3)}, {round(valid_ratio, 3)})', fontsize=20)
    plt.ylabel(y_name)
    plt.xlabel(x_name)
    plt.grid()
    plt.savefig(f'{dirname}/{name}')


distribution_df = pd.DataFrame(columns=['k', 'k_h', 'k_t', 'g_h', 'g_t'])

output_folder = 'stage7'
method = 'ComplEx'

def draw_distribution(dataset):
    dirname = f'{output_folder}/{method}_{dataset}-coef'
    distribution_dic = {}
    if not os.path.exists(dirname):
        return

    print(f'=======================merge all the csv files in {dirname}=======================')
    all_csv_files = []
    for folder in os.listdir(dirname):
        if not os.path.isdir(os.path.join(dirname, folder)):
            continue
        for file in os.listdir(os.path.join(dirname, folder)):
            if file.endswith('.csv'):
                all_csv_files.append(os.path.join(dirname, folder, file))
    print(f'len(all_csv_files): {len(all_csv_files)}')
    # df = pd.concat([pd.read_csv(x) for x in all_csv_files])

        # combine all files in the list
    file_dic = defaultdict(list)
    for filename in all_csv_files:
        file_dic[filename.split('/')[-1]].append(filename)

    for prefix, filenames in file_dic.items():
        df_list = []
        for f in filenames:
            try:
                df_list.append(pd.read_csv(f))
            except:
                print(f'cannot read {f}')
        combined_csv = pd.concat(df_list, ignore_index=True)
        combined_csv.to_csv(f"{dirname}/{prefix}", index=False, encoding='utf-8-sig')


    def float2scientific(x):
        return float('%.2E' % x)


    ###############################################################3
    print('===================plotting distribution k=================')
    # p = 'inf'
    # p = '1'
    p = '2'
    df = pd.read_csv(f'{dirname}/rel_df.csv')
    print('df.shape: ', df.shape)
    quantile = 5


    df['Delta'] = df[f'partial_{p}'] * df[f'delta_h_{p}'] * df[f'delta_t_{p}']
    df['diff'] = df['rel'] - df['head_rel'] - df['tail_rel']
    df['sum_t_h'] = df['head_rel'] + df['tail_rel']

    df['Delta_h'] = df[f'partial_t_{p}'] * df[f'delta_t_{p}']
    df['Delta_t'] = df[f'partial_h_{p}'] * df[f'delta_h_{p}']
    df['diff_h'] = df['rel'] - df['head_rel']
    df['diff_t'] = df['rel'] - df['tail_rel']   

    df['xi'] = df['diff'] / df['Delta']
    df['xi_h'] = df['diff_h'] / df['Delta_h']
    df['xi_t'] = df['diff_t'] / df['Delta_t']

    metric = pd.DataFrame()
    for suf in ['', '_h', '_t']:
        y_name = 'rel - head_rel - tail_rel' if suf == '' else f'rel - rel{suf}'
        plot(df[f'Delta{suf}'], df[f'diff{suf}'], f'draw_k_{suf}.png', x_name=f'partial{suf} * Delta{suf}', y_name=y_name, dirname=dirname)
        # plot_k(df[f'Delta{suf}'], df[f'diff{suf}'], f'ratio_k{suf}.png', x_name=f'k', y_name=f'ratio{suf}', dirname=dirname)

        # t = plot_d(df[f'xi{suf}'].values, f'draw_d{suf}.png', f'xi{suf}', f'cnt{suf}', dirname=dirname)
        # update_df(metric, t, f'{dirname}/metric.csv')

        q = calculate_quantile(df[f'xi{suf}'].values, 100 - quantile)
        print(f'k{suf}: {float2scientific(q)}')
        distribution_dic[f'k{suf}'] = float2scientific(q)


    ###############################################################
    print('===================plotting distribution g=================')
    df = pd.read_csv(f'{dirname}/hops.csv')
    print('df.shape: ', df.shape)

    metric = pd.DataFrame()
    for perspective in ['head', 'tail']:
        df_per = df[df['perspective'] == perspective].copy()
        lab = perspective[0]

        x = df_per[f'delta_all_{p}'] *  df_per[f'partial_{lab}_all_{p}']
        y = df_per['all_rel'] - df_per['rel'] 
        eta = y / x

        # print(df_per['delta'], df_per['diff'])

        plot(x, y, f'draw_g_{perspective}.png', x_name=f'delta_{lab} * partial_{lab}', y_name='all_rel - rel', dirname=dirname)
        # plot_k(x, y, f'draw_k_{perspective}.png', x_name=f'g_{perspective}', y_name=f'ratio_{perspective}', dirname=dirname)

        # t = plot_d(eta.values, f'draw_d_{lab}.png', f'eta_{lab}', f'cnt_{lab}', dirname=dirname)
        # update_df(metric, t, f'{dirname}/metric.csv')

        q = calculate_quantile(eta.values, quantile)
        print(f'g_{lab}: {float2scientific(q)}')
        distribution_dic[f'g_{lab}'] = float2scientific(q)
    distribution_df.loc[dataset] = distribution_dic


datasets = ['FB15k-237', 'WN18RR', 'MOF-3000'] # 'YAGO3-10', 

for dataset in datasets:
    draw_distribution(dataset)


print(distribution_df)
distribution_df.to_csv(f'{output_folder}/{method}-distribution.csv', encoding='utf-8-sig')



# read config.yaml and replace value coef for each dataset with distribution_df
# the coef have 5 values: k, k_h, k_t, g_h, g_t
# import yaml
# with open('config.yaml') as f:
#     config = yaml.load(f, Loader=yaml.FullLoader)
#     print(config)

# for dataset in datasets:
#     for coef in ['k', 'k_h', 'k_t', 'g_h', 'g_t']:
#         v = distribution_df.loc[dataset][coef]
#         if not float(v):
#             v = -1e-4 if coef[0] == 'g' else 1e-4
#         config[dataset]['coef'][coef] = v

# print(config)

# with open('config.yaml', 'w') as f:
#     yaml.dump(config, f)
