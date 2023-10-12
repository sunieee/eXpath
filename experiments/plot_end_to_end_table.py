import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import itertools

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

sys.path.append(
    os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir)))

parser.add_argument("--mode",
                    type=str,
                    choices=['necessary', 'sufficient'],
                    default='necessary',
                    help="The mode for which to plot the explanation lengths: necessary or sufficient")

parser.add_argument("--save",
                    type=bool,
                    default=False,
                    help="Whether to just show the table or save it in Kelpie/reproducibility_images")

args = parser.parse_args()
list1 = ['ComplEx', 'ConvE'] 
list2 = ['FB15k237', 'WN18RR', 'MOF-3000']
list3 = ['H@1', 'MRR']
df = pd.DataFrame(columns=['.'.join(t) for t in itertools.product(list1, list2, list3)])

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

def read_necessary_output_end_to_end(filepath):
    fact_to_explain_2_details = {}
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

            fact_to_explain_2_details[_fact_to_explain] = (
                _explanation_facts, _original_score, _new_score, _original_tail_rank, _new_tail_rank)

    return fact_to_explain_2_details


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


KELPIE_ROOT = os.path.realpath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir, os.pardir, os.pardir))
END_TO_END_EXPERIMENT_ROOT = os.path.realpath(os.path.join(os.path.realpath(__file__), os.pardir))

_systems = ["Criage", "DP", "K1", "Kelpie", "Xrule", "K1+Xrule", "Kelpie+Xrule"]
systems = []
for suffix in ['', '(top5)']:
    systems += [x+suffix for x in _systems]

models = ["ComplEx", "ConvE"]
datasets = ["FB15k237", "WN18RR", "MOF-3000"] # "FB15k", "WN18", 
mode = args.mode
save = args.save

output_data = []
row_labels = []
for system in systems:
    row_labels.append(f'{system}')
    system = system.replace("DP", "data_poisoning").lower()
    new_data_row = []
    for model in models:
        if model == "TransE" and system == "Criage":
            for dataset in datasets:
                new_data_row.append("-")
            continue
        
        for dataset in datasets:
            print(f"Processing {system} {model} {dataset} {mode}")
            
            end_to_end_output_filename = "_".join(
                [system, mode.lower(), model.lower(), dataset.lower().replace("-", "")]) + ".csv"
            end_to_end_output_filepath = os.path.join(END_TO_END_EXPERIMENT_ROOT, end_to_end_output_filename)
            
            # 如果有，则用新的，否则用旧的
            if system.count('(top') == 0:
                end_to_end_output_filepath = end_to_end_output_filepath.replace(system, system + '(top1)')
                # if os.path.exists(top1_file):
                #     end_to_end_output_filepath = top1_file
            
            if not os.path.isfile(end_to_end_output_filepath):
                new_data_row.append("-")
                continue

            original_ranks = []
            new_ranks = []

            # if system.startswith('Xrule'):
            try:
                prefix = ''
                fact_2_kelpie_explanations = read_xrule_output_end_to_end(end_to_end_output_filepath)
            except Exception as e:
                print(e)
                new_data_row.append("-")
                df.loc[system, f'{model}.{dataset}.H@1'] = "-"
                df.loc[system, f'{model}.{dataset}.MRR'] = "-"
                continue

            # except:
            #     prefix = '['
            #     fact_2_kelpie_explanations = read_necessary_output_end_to_end(end_to_end_output_filepath)

            for fact_to_explain in fact_2_kelpie_explanations:
                kelpie_expl, _, _, kelpie_original_tail_rank, kelpie_new_tail_rank, exp_length = fact_2_kelpie_explanations[
                    fact_to_explain]

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

            new_data_row.append(prefix + ret)

            df.loc[system, f'{model}.{dataset}.H@1'] = - round(h1_difference, 3)
            df.loc[system, f'{model}.{dataset}.MRR'] = - round(mrr_difference, 3)
    output_data.append(new_data_row)

column_labels = []


for model in models:
    for dataset in datasets:
        column_labels.append(f'{model}\n{dataset}')

fig = plt.figure(figsize=(10, 10))
ax = fig.gca()
ax.axis('off')
table = ax.table(cellText=output_data,
                 loc="center",
                 rowLoc='center',
                 cellLoc='center',
                 colLabels=column_labels,
                 rowLabels=row_labels)

if not save:
    plt.show()
else:
    table.scale(1, 1.7)
    plt.subplots_adjust(left=0.15, bottom=0.15)
    output_folder = os.path.join(KELPIE_ROOT, "reproducibility_images")
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, f'end_to_end_table_{args.mode}.png')
    print(f'Saving {args.mode} end-to-end results in {output_path}... ')
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print('Done\n')

    df.to_csv(os.path.join(output_folder, 'end_to_end_table.csv'))