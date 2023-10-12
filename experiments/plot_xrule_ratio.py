import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt

KELPIE_ROOT = os.path.realpath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir, os.pardir, os.pardir))
END_TO_END_EXPERIMENT_ROOT = os.path.realpath(os.path.join(os.path.realpath(__file__), os.pardir))

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

parser.add_argument("--save",
                    type=bool,
                    default=False,
                    help="Whether to just show the table or save it in Kelpie/reproducibility_images")
args = parser.parse_args()

def read_xrule_output_end_to_end(filepath):
    df = pd.read_csv(filepath)

    score_deteriorition  = {}
    rank_deteriorition = {}

    for i in range(len(df)):
        _explanation_facts = eval(df.loc[i, "facts"])
        _original_score, _new_score = float(df.loc[i, "original_score"]), float(df.loc[i, "new_score"])
        _original_tail_rank, _new_tail_rank = float(df.loc[i, "original_rank"]), float(df.loc[i, "new_rank"])
        _fact_to_explain = df.loc[i, "prediction"]

        # print(_explanation_facts)
        score_deteriorition[_fact_to_explain] = _original_score - _new_score
        rank_deteriorition[_fact_to_explain] = _new_tail_rank - _original_tail_rank
    return score_deteriorition, rank_deteriorition


systems = ["K1+Xrule", "Kelpie+Xrule"]
systems += [x+'(top3)' for x in systems]
models = ["ComplEx", "ConvE"]
datasets = ["FB15k237", "WN18RR", "MOF-3000"]

def update_df(df, dic):
    for key in set(dic.keys()) - set(df.columns):
        df[key] = None
    df.loc[len(df)] = dic

output_rows = []
for model in models:
    new_row = []
    for dataset in datasets:
        system2score_deteriorition = pd.DataFrame()
        system2rank_deteriorition = pd.DataFrame()
        for system in systems:
            end_to_end_output_filename = "_".join(
                        [system.lower(), 'necessary', model.lower(), dataset.lower().replace("-", "")]) + ".csv"
            end_to_end_output_filepath = os.path.join(END_TO_END_EXPERIMENT_ROOT, end_to_end_output_filename)

            if os.path.exists(end_to_end_output_filepath):
                print(f"Reading {end_to_end_output_filepath}...")
                score_deteriorition, rank_deteriorition = read_xrule_output_end_to_end(end_to_end_output_filepath)
                update_df(system2score_deteriorition, score_deteriorition)
                update_df(system2rank_deteriorition, rank_deteriorition)
        
        system2score_deteriorition = system2score_deteriorition.T
        system2rank_deteriorition = system2rank_deteriorition.T

        if len(system2score_deteriorition) > 0:
            # delete the rows with more than one column with highest value
            for ix in system2score_deteriorition.index:
                max_value = system2score_deteriorition.loc[ix].max()
                if sum(system2score_deteriorition.loc[ix] == max_value) > 1:
                    print(f'Deleting {ix} with multiple max value: {system2score_deteriorition.loc[ix]}')
                    system2score_deteriorition.drop(ix, inplace=True)
                    system2rank_deteriorition.drop(ix, inplace=True)
            
            # for each columne, select the row index with the highest value
            # for each row index, get the number of columnes with highest value
            
            # print(system2score_deteriorition)
            # print(system2score_deteriorition.idxmax(axis=1))
            out = system2score_deteriorition.idxmax(axis=1).value_counts().to_dict()
            print(out)
            if 0 not in out:
                out[0] = 0
            if 1 not in out:
                out[1] = 0
            if 2 not in out:
                out[2] = 0
            out_str = f'{out[0]}/{out[1]}/{out[2]}'
            new_row.append(out_str)
        else:
            new_row.append('-')

    output_rows.append(new_row)

fig = plt.figure(figsize=(9, 3))
ax = fig.gca()
ax.axis('off')
table = ax.table(cellText=output_rows,
                 loc="center",
                 rowLoc='center',
                 cellLoc='center',
                 colLabels=datasets,
                 rowLabels=models)


if not args.save:
    plt.show()
else:
    table.scale(1, 1.7)
    plt.subplots_adjust(left=0.15, bottom=0.15)
    output_reproducibility_folder = os.path.join(KELPIE_ROOT, "reproducibility_images")
    if not os.path.isdir(output_reproducibility_folder):
        os.makedirs(output_reproducibility_folder)
    output_path = os.path.join(output_reproducibility_folder, f'xrule_ratio_table.png')
    print(f'Saving xrule ratio in {output_path}... ')
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print('Done\n')