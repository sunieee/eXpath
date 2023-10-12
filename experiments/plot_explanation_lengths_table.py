import os
import sys
import argparse
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import itertools

sys.path.append(
    os.path.realpath(os.path.join(os.path.abspath(__file__), os.path.pardir, os.path.pardir, os.path.pardir, os.path.pardir)))


COMPLEX = "ComplEx"
CONVE = "ConvE"
TRANSE = "TransE"

COLOR_1 = "#2196f3"
COLOR_2 = "#8bc34a"
COLOR_3 = "#ffc107"
COLOR_4 = "#f44336"

list1 = ['ComplEx', 'ConvE'] 
list2 = ['FB15k237', 'WN18RR', 'MOF-3000']
df = pd.DataFrame(columns=['.'.join(t) for t in itertools.product(list1, list2)])

def read_necessary_output_end_to_end(filepath):
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
            _explanation_length = len(_explanation_facts)
            prediction_2_explanation_length[_fact_to_explain] = _explanation_length

    return prediction_2_explanation_length


def rd(x):
    return round(x, 1)

def tostr(x):
    # round to 1 decimal places and convert to string, if no decimal places, add .0
    x = rd(x)
    x = str(x)
    if "." not in x:
        x += ".0"
    return x


def read_sufficient_output_end_to_end(filepath):
    prediction_2_explanation_length = {}
    fact_to_convert_2_original_fact_to_explain = {}
    with open(filepath, "r") as input_file:
        input_lines = input_file.readlines()
        for line in input_lines:
            bits = line.strip().split(";")
            _head_to_explain, _rel_to_explain, _tail_to_explain = bits[0:3]
            _head_to_convert, _rel_to_convert, _tail_to_convert = bits[3:6]

            _fact_to_explain = (_head_to_explain, _rel_to_explain, _tail_to_explain)
            _fact_to_convert = (_head_to_convert, _rel_to_convert, _tail_to_convert)

            _explanation_bits = bits[6:-4]
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

            _explanation_length = len(_explanation_facts)
            prediction_2_explanation_length[_fact_to_convert] = _explanation_length
    return prediction_2_explanation_length


def is_triple(fact):
    return len(fact) == 3 and type(fact) == tuple and type(fact[0]) in [str, int]

def read_xrule_output_end_to_end(filepath):
    df = pd.read_csv(filepath)
    top_n = 100
    if filepath.count('+xrule'):
        top_n = int(filepath.split(')')[0].split('(top')[-1])

    prediction_2_explanation_length = {}

    for i in range(len(df)):
        _explanation_facts = eval(df.loc[i, "facts"])
        _original_score, _new_score = float(df.loc[i, "original_score"]), float(df.loc[i, "new_score"])
        _original_tail_rank, _new_tail_rank = float(df.loc[i, "original_rank"]), float(df.loc[i, "new_rank"])
        _fact_to_explain = eval(df.loc[i, "prediction"])

        # print(_explanation_facts)
        length = 0
        prediction_2_explanation_length[_fact_to_explain] = df.loc[i, "exp_length"]
    return prediction_2_explanation_length

KELPIE_ROOT = os.path.realpath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir, os.pardir, os.pardir))
END_TO_END_EXPERIMENT_ROOT = os.path.realpath(os.path.join(os.path.realpath(__file__), os.pardir))


def extract_lengths(lengths):
    output = {1:0, 2:0, 3:0, 4:0}
    for length in lengths:
        output[length] += 1
    return output

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

parser.add_argument("--save",
                    type=bool,
                    default=False,
                    help="Whether to just show the table or save it in Kelpie/reproducibility_images")
args = parser.parse_args()
mode = 'necessary'

_systems = ["K1", "Kelpie", "Xrule", "K1+Xrule", "Kelpie+Xrule"]
systems = []
for suffix in ['','(top5)']:    # , '(top5)', '(top10)'
    systems += [x+suffix for x in _systems]

models = ["ComplEx", "ConvE"]
datasets = ["FB15k237", "WN18RR", "MOF-3000"]

counts = {}
output_rows = []
for system in systems:
    system = system.replace("DP", "data_poisoning").lower()
    counts[system] = {}
    new_row = []

    for model in models:
        counts[system][model] = {}
        for dataset in datasets:
            end_to_end_output_filename = "_".join(
                    [system.lower(), mode.lower(), model.lower(), dataset.lower().replace("-", "")]) + ".csv"
            end_to_end_output_filepath = os.path.join(END_TO_END_EXPERIMENT_ROOT, end_to_end_output_filename)
            
            # 如果有，则用新的，否则用旧的
            if system.count('(top') == 0:
                end_to_end_output_filepath = end_to_end_output_filepath.replace(system, system + '(top1)')
                # if os.path.exists(top1_file):
                #     end_to_end_output_filepath = top1_file
            
            print(f"Reading {end_to_end_output_filepath}...")
            try:
                prefix = ''
                fact_2_explanation_lengths = read_xrule_output_end_to_end(end_to_end_output_filepath)
            # except:
            #     prefix = '['
            #     fact_2_explanation_lengths = read_necessary_output_end_to_end(end_to_end_output_filepath)
                cur_lengths = list(fact_2_explanation_lengths.values())
                counts[system][model][dataset] = cur_lengths
                # if tostr(numpy.average(cur_lengths)) + '±' + tostr(numpy.std(cur_lengths)) == '0.0±0.4':
                #     print(cur_lengths)
                new_row.append(prefix + tostr(numpy.average(cur_lengths))) # + '±' + tostr(numpy.std(cur_lengths)))

                df.loc[system, f'{model}.{dataset}'] = tostr(numpy.average(cur_lengths))  # + '±' + tostr(numpy.std(cur_lengths))
            except:
                ret = '-'
                new_row.append(ret)
                df.loc[system, f'{model}.{dataset}'] = ret


    output_rows.append(new_row)

column_labels = []

for model in models:
    for dataset in datasets:
        column_labels.append(f'{model}\n{dataset}')

fig = plt.figure(figsize=(9, 9))
ax = fig.gca()
ax.axis('off')
table = ax.table(cellText=output_rows,
                 loc="center",
                 rowLoc='center',
                 cellLoc='center',
                 colLabels=column_labels,
                 rowLabels=systems)

if not args.save:
    plt.show()
else:
    table.scale(1, 1.7)
    plt.subplots_adjust(left=0.15, bottom=0.15)
    output_folder = os.path.join(KELPIE_ROOT, "reproducibility_images")
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    output_path = os.path.join(output_folder, f'explanation_lengths_table_{mode}.png')
    print(f'Saving {mode} explanation lengths in {output_path}... ')
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print('Done\n')

    df.to_csv(os.path.join(output_folder, f'explanation_lengths_table.csv'))