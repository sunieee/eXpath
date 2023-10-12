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
        kelpie = Kelpie(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=prefilter,
                        relevance_threshold=relevance_threshold)
    elif args.system == "data_poisoning":
        kelpie = DataPoisoning(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=prefilter)
    elif args.system == "criage":
        kelpie = Criage(model=model, dataset=dataset, hyperparameters=hyperparameters)
    elif args.system == "k1":
        kelpie = Kelpie(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=prefilter,
                        relevance_threshold=relevance_threshold, max_explanation_length=1)
    else:
        kelpie = Kelpie(model=model, dataset=dataset, hyperparameters=hyperparameters, prefilter_type=prefilter,
                        relevance_threshold=relevance_threshold)


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
    head, relation, tail = prediction
    i = testing_samples.index(prediction)
    ech(f"Explaining fact {i}/{len(testing_samples)}: {prediction}  {dataset.sample_to_fact(prediction, True)}")
    # log(dataset.entity_name_2_id)
    score, best, rank = extract_performances(model, prediction)
    
    ##########################################################
    # generate explanations even if the prediction rank > 1
    # if rank > 1:
    #     log(f'{dataset.sample_to_fact(prediction, True)} is not a valid prediction (rank={rank}, score={score}). Skip')
    #     continue
    log(f'rank: {rank}')
    
    # path_statistic(prediction)
    paths = dataset.find_all_path_within_k_hop(head, tail, 3)
    # 过滤掉长度为1的路径，这种路径是一种简单的推导，不能作为解释
    paths = [p for p in paths if len(p) > 1]
    super_paths = set()
    available_samples = defaultdict(set)
    phs = {}
    pts = {}
    for p in paths:
        super = get_path_entities(prediction, p)
        if super[1] in phs:
            phs[super[1]] = min(len(super) - 1, phs[super[1]])
        else:
            phs[super[1]] = len(super) - 1
        if super[-2] in pts:
            pts[super[-2]] = min(len(super) - 1, pts[super[-2]])
        else:
            pts[super[-2]] = len(super) - 1
        super_paths.add(tuple(super))
        for t in p:
            # print(t, t[0], t[2])
            available_samples[t[0]].add(t)
            available_samples[t[2]].add(t)
    ech(f'all related entities on path: {len(available_samples)}, #phs: {len(phs)}, #pts: {len(pts)}, #paths: {len(paths)}, #super_paths: {len(super_paths)}')
    # random_explain_path(super_paths)
    # random_explain_group(phs, pts, prediction)
    # explain_all_path(super_paths, prediction, phs, pts)

    ech('Creating generators')
    head_generator = OneHopGenerator('head', prediction, phs, available_samples=available_samples)
    tail_generator = OneHopGenerator('tail', prediction, pts, available_samples=available_samples)
    path_generator = PathGenerator(prediction, super_paths, available_samples=available_samples)

    ech('Start generating...')
    while not head_generator.finished():
        pair = head_generator.generate()
        if pair is not None:
            path_generator.renew_head(pair[0], pair[1])
    head_generator.close()

    while not tail_generator.finished():
        pair = tail_generator.generate()
        if pair is not None:
            path_generator.renew_tail(pair[0], pair[1])
    tail_generator.close()

    while not path_generator.finished():
        pair = path_generator.generate()
    path_generator.close()


def explain_one(func):
    for sample in testing_samples:
        if str(sample) not in os.listdir(args.already_explain_path):
            open(f'{args.already_explain_path}/{sample}', 'w').close()
            func(sample)
            return True
    return False


if args.system == 'xrule':
    func = explain_sample
else: 
    func = explain_sample_kelpie


if args.process > 1:
    ech(f'Splitting testing facts: {args.split}/{args.process}')
    cnt = 0
    while explain_one(func):
        cnt += 1 
        ech(f'Explained {cnt} facts')
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
    

'''
RuntimeError: CUDA error: an illegal memory access was encountered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.


File "/home/sy/2023/Kelpie-copy/utils.py", line 560, in build
    explanation.calculate_relevance()
  File "/home/sy/2023/Kelpie-copy/utils.py", line 625, in calculate_relevance
    pt_embeddings = pt_training()
  File "/home/sy/2023/Kelpie-copy/utils.py", line 713, in pt_training_multiple
    return self.post_training_multiple(self.pt_train_samples, early_stop=True)
  File "/home/sy/2023/Kelpie-copy/utils.py", line 721, in post_training_multiple
    results.append(self.post_training_save(training_samples))
  File "/home/sy/2023/Kelpie-copy/utils.py", line 796, in post_training_save
    optimizer.train(train_samples=post_train_samples, post_train=True)
  File "/home/sy/2023/Kelpie-copy/link_prediction/optimization/bce_optimizer.py", line 81, in train
    self.epoch(er_vocab=er_vocab, er_vocab_pairs=er_vocab_pairs, batch_size=self.batch_size, post_train=post_train)
  File "/home/sy/2023/Kelpie-copy/link_prediction/optimization/bce_optimizer.py", line 130, in epoch
    batch, targets = self.extract_batch(er_vocab=er_vocab,
  File "/home/sy/2023/Kelpie-copy/link_prediction/optimization/bce_optimizer.py", line 113, in extract_batch
    return torch.tensor(batch).cuda(), torch.FloatTensor(targets).cuda()
RuntimeError: CUDA error: an illegal memory access was encountered
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.


  File "/home/sy/2023/Kelpie-copy/link_prediction/optimization/bce_optimizer.py", line 81, in train
    self.epoch(er_vocab=er_vocab, er_vocab_pairs=er_vocab_pairs, batch_size=self.batch_size, post_train=post_train)
  File "/home/sy/2023/Kelpie-copy/link_prediction/optimization/bce_optimizer.py", line 141, in epoch
    l = self.step_on_batch(batch, targets, post_train=post_train)
  File "/home/sy/2023/Kelpie-copy/link_prediction/optimization/bce_optimizer.py", line 185, in step_on_batch
    loss.backward()
  File "/home/sy/anaconda3/envs/torch/lib/python3.9/site-packages/torch/_tensor.py", line 487, in backward
    torch.autograd.backward(
  File "/home/sy/anaconda3/envs/torch/lib/python3.9/site-packages/torch/autograd/__init__.py", line 200, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
RuntimeError: CUDA error: an illegal memory access was encountered
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

pkill -f "explain.py"

{'MOF-3000': {'tail_restrain': {'hasColor': 'M0->C0', 'hasHabit': 'M0->H0', 'hasPaper': 'M0->P0', 'hasJournal': 'M0->J0', 'hasAuthor': 'M0->A0', 'hasKernel': 'M0,P1,B0->K1,K2,K3', 'subKernel': 'K2,K3->K1,K2', 'hasBond': 'M0,P1->B0', 'hasSolvent': 'M0->ref', 'hasLinker': 'M0->ref', 'hasMetal': 'M0->ref', 'hasOperation': 'M0->O0', 'hasMethod': 'M0->M1'}, 'TransE': {'D': 50, 'LR': 0.0002, 'B': 4096, 'Ep': 200, 'gamma': 2, 'N': 5, 'Opt': 'Adam', 'Reg': 0}, 'ComplEx': {'D': 500, 'LR': 0.1, 'B': 2000, 'Ep': 50, 'Opt': 'Adagrad', 'Reg': 0.05}, 'ConvE': {'D': 200, 'LR': 0.003, 'B': 256, 'Ep': 50, 'Decay': 0.995, 'epsilon': 0.1, 'Drop': {'in': 0.2, 'h': 0.3, 'feat': 0.2}}, 'coef': {'k_h': 0.00096, 'k_t': 0, 'k': 0.0083, 'g_h': -0.036, 'g_t': -0.011}}, 'FB15k-237': {'TransE': {'D': 50, 'LR': 0.0004, 'B': 2048, 'Ep': 100, 'gamma': 5, 'N': 15, 'Opt': 'Adam', 'Reg': 1}, 'ComplEx': {'D': 1000, 'LR': 0.1, 'B': 1000, 'Ep': 50, 'Opt': 'Adagrad', 'Reg': 0.05}, 'ConvE': {'D': 200, 'LR': 0.003, 'B': 128, 'Ep': 50, 'Decay': 0.995, 'epsilon': 0.1, 'Drop': {'in': 0.2, 'h': 0.3, 'feat': 0.2}}, 'coef': {'k_h': 0.026, 'k_t': 6.9e-05, 'k': 0.11, 'g_h': -0.029, 'g_t': -0.011}}, 'WN18': {'TransE': {'D': 50, 'LR': 0.0002, 'B': 2048, 'Ep': 200, 'gamma': 2, 'N': 5, 'Opt': 'Adam', 'Reg': 0}, 'ComplEx': {'D': 500, 'LR': 0.1, 'B': 1000, 'Ep': 50, 'Opt': 'Adagrad', 'Reg': 0.05}, 'ConvE': {'D': 200, 'LR': 0.003, 'B': 128, 'Ep': 50, 'Decay': 0.995, 'epsilon': 0.1, 'Drop': {'in': 0.2, 'h': 0.3, 'feat': 0.2}}, 'coef': {'k_h': 0.13, 'k_t': 0.017, 'k': 0.51, 'g_h': -0.064, 'g_t': -0.011}}, 'YAGO3-10': {'TransE': {'D': 200, 'LR': 0.0001, 'B': 2048, 'Ep': 100, 'gamma': 5, 'N': 5, 'Opt': 'Adam', 'Reg': 50}, 'ComplEx': {'D': 1000, 'LR': 0.1, 'B': 1000, 'Ep': 50, 'Opt': 'Adagrad', 'Reg': 0.005}, 'ConvE': {'D': 200, 'LR': 0.003, 'B': 128, 'Ep': 20, 'Decay': 0.995, 'epsilon': 0.1, 'Drop': {'in': 0.2, 'h': 0.3, 'feat': 0.2}}, 'coef': {'k_h': 0.2, 'k_t': 0.0014, 'k': 0.76, 'g_h': -0.011, 'g_t': -0.011}}}
{'MOF-3000': {'tail_restrain': {'hasColor': 'M0->C0', 'hasHabit': 'M0->H0', 'hasPaper': 'M0->P0', 'hasJournal': 'M0->J0', 'hasAuthor': 'M0->A0', 'hasKernel': 'M0,P1,B0->K1,K2,K3', 'subKernel': 'K2,K3->K1,K2', 'hasBond': 'M0,P1->B0', 'hasSolvent': 'M0->ref', 'hasLinker': 'M0->ref', 'hasMetal': 'M0->ref', 'hasOperation': 'M0->O0', 'hasMethod': 'M0->M1'}, 'TransE': {'D': 50, 'LR': 0.0002, 'B': 4096, 'Ep': 200, 'gamma': 2, 'N': 5, 'Opt': 'Adam', 'Reg': 0}, 'ComplEx': {'D': 500, 'LR': 0.1, 'B': 2000, 'Ep': 50, 'Opt': 'Adagrad', 'Reg': 0.05}, 'ConvE': {'D': 200, 'LR': 0.003, 'B': 256, 'Ep': 50, 'Decay': 0.995, 'epsilon': 0.1, 'Drop': {'in': 0.2, 'h': 0.3, 'feat': 0.2}}, 'coef': {'k_h': 0.0425, 'k_t': 0.000841, 'k': 1.33, 'g_h': -0.0269, 'g_t': -0.0001}}, 'FB15k-237': {'TransE': {'D': 50, 'LR': 0.0004, 'B': 2048, 'Ep': 100, 'gamma': 5, 'N': 15, 'Opt': 'Adam', 'Reg': 1}, 'ComplEx': {'D': 1000, 'LR': 0.1, 'B': 1000, 'Ep': 50, 'Opt': 'Adagrad', 'Reg': 0.05}, 'ConvE': {'D': 200, 'LR': 0.003, 'B': 128, 'Ep': 50, 'Decay': 0.995, 'epsilon': 0.1, 'Drop': {'in': 0.2, 'h': 0.3, 'feat': 0.2}}, 'coef': {'k_h': 0.186, 'k_t': 0.0248, 'k': 3.6, 'g_h': -0.0251, 'g_t': -0.0309}}, 'WN18': {'TransE': {'D': 50, 'LR': 0.0002, 'B': 2048, 'Ep': 200, 'gamma': 2, 'N': 5, 'Opt': 'Adam', 'Reg': 0}, 'ComplEx': {'D': 500, 'LR': 0.1, 'B': 1000, 'Ep': 50, 'Opt': 'Adagrad', 'Reg': 0.05}, 'ConvE': {'D': 200, 'LR': 0.003, 'B': 128, 'Ep': 50, 'Decay': 0.995, 'epsilon': 0.1, 'Drop': {'in': 0.2, 'h': 0.3, 'feat': 0.2}}, 'coef': {'k_h': 0.804, 'k_t': 0.0639, 'k': 5.66, 'g_h': -0.00115, 'g_t': -0.0888}}, 'YAGO3-10': {'TransE': {'D': 200, 'LR': 0.0001, 'B': 2048, 'Ep': 100, 'gamma': 5, 'N': 5, 'Opt': 'Adam', 'Reg': 50}, 'ComplEx': {'D': 1000, 'LR': 0.1, 'B': 1000, 'Ep': 50, 'Opt': 'Adagrad', 'Reg': 0.005}, 'ConvE': {'D': 200, 'LR': 0.003, 'B': 128, 'Ep': 20, 'Decay': 0.995, 'epsilon': 0.1, 'Drop': {'in': 0.2, 'h': 0.3, 'feat': 0.2}}, 'coef': {'k_h': 0.0731, 'k_t': 0.0116, 'k': 1.54, 'g_h': -0.0001, 'g_t': -0.0187}}}
'''