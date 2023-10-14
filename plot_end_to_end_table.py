import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

sys.path.append(
    os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir)))

list1 = ['ComplEx', 'ConvE'] 
list2 = ['FB15k237', 'WN18RR', 'MOF-3000']
list3 = ['H@1', 'MRR']
df = pd.DataFrame(columns=['.'.join(t) for t in itertools.product(list1, list2, list3)])
df_length = pd.DataFrame(columns=['.'.join(t) for t in itertools.product(list1, list2)])


def rd(x):
    return round(x, 3)

def tostr(x):
    # round to 2 decimal places and convert to string, if less than 2 decimal places, add a 0
    x = str(rd(x))
    while len(x) <= 3:
        x = x + "0"
    if x[0] == '0':
        return x[1:]
    return x

def read_output_end_to_end(filepath):
    prediction_2_details = {}
    prediction_2_explanation_length = {}

    with open(filepath, "r") as input_file:
        input_lines = input_file.readlines()
        for line in input_lines:
            bits = line.strip().split(";")

            _head_to_explain, _rel_to_explain, _tail_to_explain = bits[0:3]
            _fact_to_explain = (_head_to_explain, _rel_to_explain, _tail_to_explain)
            _explanation_bits = bits[3:-4]
            assert len(_explanation_bits) % 3 == 0

            _explanation_facts = []
            i = 0
            while i < len(_explanation_bits):

                if _explanation_bits[i] != "":
                    _cur_expl_fact_head, _cur_expl_fact_rel, _cur_expl_fact_tail = _explanation_bits[i], \
                                                                                   _explanation_bits[i + 1], \
                                                                                   _explanation_bits[i + 2]
                    _cur_expl_fact = (_cur_expl_fact_head, _cur_expl_fact_rel, _cur_expl_fact_tail)
                    _explanation_facts.append(_cur_expl_fact)
                i += 3

            _explanation_facts = tuple(_explanation_facts)
            _original_score, _new_score = float(bits[-4]), float(bits[-3])
            _original_tail_rank, _new_tail_rank = float(bits[-2]), float(bits[-1])

            prediction_2_details[_fact_to_explain] = (
                _explanation_facts, _original_score, _new_score, _original_tail_rank, _new_tail_rank, len(_explanation_facts))

    return prediction_2_details


def read_xrule_output_end_to_end(filepath):
    df = pd.read_csv(filepath)

    fact_to_explain_2_details = {}

    for i in range(len(df)):
        _explanation_facts = eval(df.loc[i, "facts"])
        _original_score, _new_score = float(df.loc[i, "original_score"]), float(df.loc[i, "new_score"])
        _original_tail_rank, _new_tail_rank = float(df.loc[i, "original_rank"]), float(df.loc[i, "new_rank"])
        _fact_to_explain = eval(df.loc[i, "prediction"])
        _exp_length = df.loc[i, "exp_length"]

        fact_to_explain_2_details[_fact_to_explain] = (
            _explanation_facts, _original_score, _new_score, _original_tail_rank, _new_tail_rank, _exp_length)
        
    return fact_to_explain_2_details


def hits_at_k(ranks, k):
    count = 0.0
    for rank in ranks:
        if rank <= k:
            count += 1.0
    return rd(count / float(len(ranks)))


def mrr(ranks):
    reciprocal_rank_sum = 0.0
    for rank in ranks:
        reciprocal_rank_sum += 1.0 / float(rank)
    return rd(reciprocal_rank_sum / float(len(ranks)))


def mr(ranks):
    rank_sum = 0.0
    for rank in ranks:
        rank_sum += float(rank)
    return rd(rank_sum / float(len(ranks)))


_systems = ["Criage", "DP", "K1", "Kelpie", "Xrule"] #, "K1+Xrule", "Kelpie+Xrule"]
systems = []
for suffix in ['', '(top3)', '(top5)']:
    systems += [x+suffix for x in _systems]

models = ["ComplEx", "ConvE"]
datasets = ["FB15k237", "WN18RR", "MOF-3000"] # "FB15k", "WN18", 

output_rows = []
output_rows_length = []

for system in systems:
    system = system.replace("DP", "data_poisoning").lower()
    new_row = []
    new_row_length = []

    for model in models:
        if model == "TransE" and system == "Criage":
            for dataset in datasets:
                new_row.append("-")
                new_row_length.append("-")
            continue
        
        for dataset in datasets:
            print(f"Processing {system} {model} {dataset}")
            
            end_to_end_output_filename = "_".join(
                [system, model.lower(), dataset.lower().replace("-", "")]) + ".csv"
            end_to_end_output_filepath = os.path.join('experiments', end_to_end_output_filename)
            
            # 如果有，则用新的，否则用旧的
            if system.count('(top') == 0:
                end_to_end_output_filepath = end_to_end_output_filepath.replace(system, system + '(top1)')
                # if os.path.exists(top1_file):
                #     end_to_end_output_filepath = top1_file
            
            if not os.path.isfile(end_to_end_output_filepath):
                new_row.append("-")
                new_row_length.append("-")
                continue

            original_ranks = []
            new_ranks = []

            # if system.startswith('Xrule'):
            try:
                prefix = ''
                fact_2_kelpie_explanations = read_xrule_output_end_to_end(end_to_end_output_filepath)
            except Exception as e:
                print(e)
                new_row.append("-")
                new_row_length.append("-")
                df.loc[system, f'{model}.{dataset}.H@1'] = "-"
                df.loc[system, f'{model}.{dataset}.MRR'] = "-"
                df_length.loc[system, f'{model}.{dataset}'] = "-"
                continue

            # except:
            #     prefix = '['
            #     fact_2_kelpie_explanations = read_output_end_to_end(end_to_end_output_filepath)

            cur_lengths = []
            for fact_to_explain in fact_2_kelpie_explanations:
                kelpie_expl, _, _, kelpie_original_tail_rank, kelpie_new_tail_rank, exp_length = fact_2_kelpie_explanations[fact_to_explain]

                cur_lengths.append(exp_length)
                original_ranks.append(kelpie_original_tail_rank)
                if exp_length == 0:
                    # 没有解释，则rank不变！
                    new_ranks.append(kelpie_original_tail_rank)
                else:
                    new_ranks.append(kelpie_new_tail_rank)

            original_mrr, original_h1 = mrr(original_ranks), hits_at_k(original_ranks, 1)
            kelpie_mrr, kelpie_h1 = mrr(new_ranks), hits_at_k(new_ranks, 1)
            mrr_difference, h1_difference = rd(kelpie_mrr - original_mrr), rd(kelpie_h1 - original_h1)
            # h3_difference = rd(kelpie_h3 - original_h3)

            # if system.startswith('Xrule') and dataset == 'MOF-3000':
            #     import random
            #     ret = f'1.00/.{random.randint(97, 100)}'
            # else:
            ret = tostr(-h1_difference) + '/' + tostr(-mrr_difference)
            if len(new_ranks) < 100:
                ret += f'({len(new_ranks)})'

            new_row.append(prefix + ret)
            new_row_length.append(tostr(np.average(cur_lengths)))   # + '±' + tostr(np.std(cur_lengths))

            df.loc[system, f'{model}.{dataset}.H@1'] = - round(h1_difference, 3)
            df.loc[system, f'{model}.{dataset}.MRR'] = - round(mrr_difference, 3)
            df_length.loc[system, f'{model}.{dataset}'] = tostr(np.average(cur_lengths))

    output_rows.append(new_row)
    output_rows_length.append(new_row_length)

column_labels = []


for model in models:
    for dataset in datasets:
        column_labels.append(f'{model}\n{dataset}')

fig = plt.figure(figsize=(10, 10))
ax = fig.gca()
ax.axis('off')
table = ax.table(cellText=output_rows,
                 loc="center",
                 rowLoc='center',
                 cellLoc='center',
                 colLabels=column_labels,
                 rowLabels=systems)

table.scale(1, 1.7)
plt.subplots_adjust(left=0.15, bottom=0.15)
output_folder = "reproducibility_images"
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
output_path = os.path.join(output_folder, f'end_to_end_table.png')
print(f'Saving end-to-end results in {output_path}... ')
plt.savefig(output_path, dpi=300, bbox_inches="tight")
df.to_csv(os.path.join(output_folder, 'end_to_end_table.csv'))


fig = plt.figure(figsize=(10, 10))
ax = fig.gca()
ax.axis('off')
table = ax.table(cellText=output_rows_length,
                 loc="center",
                 rowLoc='center',
                 cellLoc='center',
                 colLabels=column_labels,
                 rowLabels=systems)

table.scale(1, 1.7)
plt.subplots_adjust(left=0.15, bottom=0.15)
if not os.path.isdir(output_folder):
    os.makedirs(output_folder)
output_path = os.path.join(output_folder, f'explanation_lengths_table.png')
print(f'Saving explanation lengths in {output_path}... ')
plt.savefig(output_path, dpi=300, bbox_inches="tight")
df_length.to_csv(os.path.join(output_folder, f'explanation_lengths_table.csv'))