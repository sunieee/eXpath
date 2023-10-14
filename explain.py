import os
import argparse
import random
import time
import numpy
import torch
import yaml
import click
import pandas as pd
import numpy as np
import math
from datetime import datetime
import re
from tqdm import tqdm

from link_prediction.models.model import *
from link_prediction.evaluation.evaluation import Evaluator
from utils import *


print(args.run, int(args.run[0]), int(args.run[1]))
# ---------------------train---------------------
if int(args.run[0]):
    ech("Training model...")
    t = time.time()
    optimizer = Optimizer(model=model, hyperparameters=hyperparameters)
    optimizer.train(train_samples=dataset.train_samples, evaluate_every=2, # if args.method == "ConvE" else -1,
                    save_path=args.model_path,
                    valid_samples=dataset.valid_samples, patience=10)
    print(f"Train time: {time.time() - t}")

# ---------------------test---------------------
def ignore_triple(triple):
    # print(triple)
    h, r, t = triple
    if r in ['hasmethod', 'hasoperation']:
        return True
    if t in ['p1962']:
        return True
    return False


model.eval()
if int(args.run[1]):
    ech("Evaluating model...")
    # typ = 'tail' if args.dataset.startswith('MOF') else 'all'
    typ = 'tail'
    Evaluator(model=model).evaluate(samples=dataset.test_samples, write_output=True, \
                                    folder=args.output_folder, type=typ)

# 必须在最后一步输出，更新文件
ech(f'finished Train & Evaluate! run:{args.run}')

# ---------------------generating predictions---------------------
if int(args.run[2]):
    ech("Generating facts to explain (prediction that ranks first)...")
    lis = []
    print("{:^30}\t{:^15}\t{:^15}\t{:^15}".format('relation', '#targets', '#triples', '#top_triples'))
    print(os.path.join(args.output_folder, 'filtered_ranks.csv'))
    df = pd.read_csv(os.path.join(args.output_folder, 'filtered_ranks.csv'), sep=';', header=None, dtype=str)
    df.columns = ['h', 'r', 't', 'hr', 'tr']
    df['hr'] = df['hr'].astype(int)
    df['tr'] = df['tr'].astype(int)

    for d in set(df['r']):
        rel_df = df[df['r'] == d]
        rel_df.reset_index(inplace=True)
        size = len(dataset.rid2target[dataset.relation_name_2_id[d]])
        top_count = 0
        for i in range(len(rel_df)):
            # if df.loc[i, 'tr'] <= math.ceil(size*0.05):
            if rel_df.loc[i, 'tr'] != 1:
                continue
            
            # make sure tr and hr are 1 except for MOF dataset
            top_count += 1
            lis.append('\t'.join([str(rel_df.loc[i, 'h']), rel_df.loc[i, 'r'], str(rel_df.loc[i, 't'])]))
        print("{:^30}\t{:^15}\t{:^15}\t{:^15}".format(d, size, len(rel_df), top_count))

    # randomly choose 100 facts to explain in lis
    random.shuffle(lis)
    lis.sort(key=lambda x: ignore_triple(x.split('\t')))
    lis = lis[:100]

    with open(args.explain_path, 'w') as f:
        f.write('\n'.join(lis))
    print(lis)

# ---------------------explain---------------------
if not int(args.run[3]):
    os._exit(1)


ech(f"Reading facts to explain... from {args.explain_path}")
with open(args.explain_path, "r") as facts_file:
    testing_facts = [x.strip().split("\t") for x in facts_file.readlines()]
    testing_samples = []
    
    for fact in testing_facts:
        h, r, t = fact
        head, relation, tail = dataset.get_id_for_entity_name(h), \
                                    dataset.get_id_for_relation_name(r), \
                                    dataset.get_id_for_entity_name(t)
        testing_samples.append((head, relation, tail))

print("len(testing_facts):", len(testing_facts))
# get the ids of the elements of the fact to explain and the perspective entity
# xrule = Xrule(model, dataset)

print('dataset size:', dataset.train_samples.shape, len(dataset.entity_id_2_name), len(dataset.relation_id_2_name))


if args.system != 'xrule':
    ech(f'explaining facts using {args.system}...')
    from kelpie import Kelpie
    from data_poisoning import DataPoisoning
    from criage import Criage
    if args.system == "kelpie":
        kelpie = Kelpie(model=model, dataset=dataset, hyperparameters=hyperparameters,
                        relevance_threshold=relevance_threshold)
    elif args.system == "data_poisoning":
        kelpie = DataPoisoning(model=model, dataset=dataset, hyperparameters=hyperparameters)
    elif args.system == "criage":
        kelpie = Criage(model=model, dataset=dataset, hyperparameters=hyperparameters)
    elif args.system == "k1":
        kelpie = Kelpie(model=model, dataset=dataset, hyperparameters=hyperparameters,
                        relevance_threshold=relevance_threshold, max_explanation_length=1)
    else:
        kelpie = Kelpie(model=model, dataset=dataset, hyperparameters=hyperparameters,
                        relevance_threshold=relevance_threshold)
else:
    kelpie_folder = re.sub(r'(/\d+)', '', args.output_folder.replace('xrule', 'kelpie'))
    kelpie_df = pd.read_csv(os.path.join(kelpie_folder, 'output_end_to_end_kelpie(top1).csv'))
    kelpie_df.set_index('prediction', inplace=True)


def explain_sample_kelpie(prediction):
    i = testing_samples.index(prediction)
    fact = dataset.sample_to_fact(prediction)
    ech(f"Explaining fact {i}/{len(testing_samples)}: {prediction}  {dataset.sample_to_fact(prediction, True)}")
    rule_samples_with_relevance = kelpie.explain_necessary(prediction,
                                                        perspective="head",
                                                        num_promising_samples=args.prefilter_threshold)
    
    output_lines = []
    rule_facts_with_relevance = []
    for cur_rule_with_relevance in rule_samples_with_relevance:
        cur_rule_samples, cur_relevance = cur_rule_with_relevance

        cur_rule_facts = [dataset.sample_to_fact(sample) for sample in cur_rule_samples]
        cur_rule_facts = ";".join([";".join(x) for x in cur_rule_facts])
        rule_facts_with_relevance.append(cur_rule_facts + ":" + str(cur_relevance))
    print(";".join(fact))
    print(", ".join(rule_facts_with_relevance))
    print()
    output_lines.append(";".join(fact) + "\n")
    output_lines.append(",".join(rule_facts_with_relevance) + "\n")
    output_lines.append("\n")

    if not os.path.exists(f"{args.output_folder}/output.txt"):
        open(f"{args.output_folder}/output.txt", "w").close()
    with open(f"{args.output_folder}/output.txt", "a") as output:
        output.writelines(output_lines)


def explain_sample(prediction):
    # if ignore_triple(fact):
    #     continue
    ech('cuda summary:')
    print(torch.cuda.memory_summary())
    Explanation.empty_cache()
    head, relation, tail = prediction
    i = testing_samples.index(prediction)
    print()
    ech(f"Explaining fact {i}/{len(testing_samples)}: {prediction}  {dataset.sample_to_fact(prediction, True)}")
    # print(dataset.entity_name_2_id)
    score, best, rank = extract_performances(model, prediction)
    
    ##########################################################
    # generate explanations even if the prediction rank > 1
    # if rank > 1:
    #     print(f'{dataset.sample_to_fact(prediction, True)} is not a valid prediction (rank={rank}, score={score}). Skip')
    #     continue
    print(f'rank: {rank}')
    
    # path_statistic(prediction)
    all_paths = dataset.find_all_path_within_k_hop(head, tail, 3)
    # 过滤掉长度为1的路径，这种路径是一种简单的推导，不能作为解释
    # 使用kelpie top20 facts过滤路径
    all_paths = [p for p in all_paths if len(p) > 1]

    facts = eval(kelpie_df.loc[str(prediction), 'facts'])
    print('kelpie facts:', facts)
    facts = sorted(facts, key=lambda x: x[1], reverse=True)
    fact2score = defaultdict(float)
    fact2mrr = defaultdict(float)
    for ix, f in enumerate(facts):
        for fact in f[0]:
            fact2score[fact] += f[1]
            fact2mrr[fact] += 1 / (ix + 1)
    
    fact2score = sorted(fact2score.items(), key=lambda x: x[1], reverse=True)
    fact2mrr = sorted(fact2mrr.items(), key=lambda x: x[1], reverse=True)

    top_facts = set([f[0] for f in fact2score[:20]] + [f[0] for f in fact2mrr[:20]])
    print('top facts:', top_facts)
    top_phs = [t if h == head else h for h, r, t in top_facts]
    print('top phs:', top_phs)

    paths = []
    hyperpaths = set()
    available_samples = defaultdict(set)
    phs = {}
    pts = {}
    ph_meta_dic = defaultdict(int)
    for p in all_paths:
        hyperpath = get_path_entities(prediction, p)
        if hyperpath[1] not in top_phs:
            continue

        meta = tuple([t[1] for t in p])
        if ph_meta_dic[(hyperpath[1], meta)] >= 2:
            continue
        ph_meta_dic[(hyperpath[1], meta)] += 1

        if hyperpath[1] in phs:
            phs[hyperpath[1]] = min(len(hyperpath) - 1, phs[hyperpath[1]])
        else:
            phs[hyperpath[1]] = len(hyperpath) - 1
        if hyperpath[-2] in pts:
            pts[hyperpath[-2]] = min(len(hyperpath) - 1, pts[hyperpath[-2]])
        else:
            pts[hyperpath[-2]] = len(hyperpath) - 1
        hyperpaths.add(hyperpath)
        paths.append(hyperpath)
        for t in p:
            # print(t, t[0], t[2])
            available_samples[t[0]].add(t)
            available_samples[t[2]].add(t)
    ech(f'all related entities on path')
    print(f'#samples: {len(available_samples)}, #phs: {len(phs)}, #pts: {len(pts)}, #all_paths: {len(all_paths)}, #paths: {len(paths)}, #hyperpaths: {len(hyperpaths)}')
    print('phs:', phs)
    print('pts:', pts)
    print('hyperpaths:', hyperpaths)
    print('available_samples:', available_samples)
    prediction2concerning_entities[prediction] = phs.keys() | pts.keys()
    print('concerning_entities:', prediction2concerning_entities[prediction])
    # random_explain_path(super_paths)
    # random_explain_group(phs, pts, prediction)
    # explain_all_path(super_paths, prediction, phs, pts)

    ech('Creating generators')
    # tail_generator = OneHopGenerator('tail', prediction, pts, available_samples=available_samples)
    path_generator = PathGenerator(prediction, hyperpaths, available_samples=available_samples)

    ech('Calculate head relevance')
    for ph in phs:
        exp = Explanation(prediction, available_samples[ph] & available_samples[head], [head])
        path_generator.renew_head(ph, exp)

    ech('Calculate tail relevance')
    # while not tail_generator.finished():
    #     pair = tail_generator.generate()
    #     if pair is not None:
    #         path_generator.renew_tail(pair[0], pair[1])
    # tail_generator.close()

    # random select 50 pts
    pts = list(pts.keys())
    random.shuffle(pts)
    pts = pts[:50]
    for pt in pts:
        exp = Explanation(prediction, available_samples[pt] & available_samples[tail], [tail])
        path_generator.renew_tail(pt, exp)

    ech('Calculate path relevance')
    while not path_generator.finished():
        pair = path_generator.generate()
    path_generator.close()

    # del tail_generator
    del path_generator

def explain_one(func, pbar=None):
    for sample in testing_samples:
        if str(sample) not in os.listdir(args.already_explain_path):
            open(f'{args.already_explain_path}/{sample}', 'w').close()
            func(sample)
            if pbar:
                pbar.n = len(os.listdir(args.already_explain_path))
                pbar.refresh()
            return True
    return False

if args.system == 'xrule':
    func = explain_sample
else: 
    func = explain_sample_kelpie


if args.process > 1:
    ech(f'Splitting testing facts: {args.split}/{args.process}')
    
    # 初始化进度条
    pbar = tqdm(total=len(testing_samples), initial=0, desc="Explaining")

    cnt = 0
    while explain_one(func, pbar):
        cnt += 1 
        ech(f'Explained {cnt} facts')
    
    pbar.close()
    ech(f'All facts explained, total: {cnt}')
else:
    for i, sample in enumerate(testing_samples):
        explain_sample(sample)


# # use thread pool (size = 10) to explain facts
# if MULTI_THREAD > 1:
#     from multiprocessing.pool import ThreadPool
#     pool = ThreadPool(MULTI_THREAD)
#     pool.map(explain_fact, testing_facts)
#     pool.close()
    
