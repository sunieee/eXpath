import os
import argparse
import random
import time
import numpy
import torch

from dataset import Dataset
import yaml
import click
import pandas as pd
import numpy as np
import math
import logging
from datetime import datetime
from dataset import Dataset
import json
from queue import PriorityQueue

from link_prediction.models.transe import TransE
from link_prediction.models.complex import ComplEx
from link_prediction.models.conve import ConvE, PostConvE
from link_prediction.models.distmult import DistMult
from link_prediction.models.gcn import CompGCN
from link_prediction.models.model import *
from link_prediction.optimization.bce_optimizer import BCEOptimizer
from link_prediction.optimization.multiclass_nll_optimizer import MultiClassNLLOptimizer
from link_prediction.optimization.pairwise_ranking_optimizer import PairwiseRankingOptimizer
from prefilters.prefilter import TOPOLOGY_PREFILTER, TYPE_PREFILTER, NO_PREFILTER
from link_prediction.models.tucker import TuckER
from link_prediction.optimization.bce_optimizer import KelpieBCEOptimizer
from link_prediction.optimization.multiclass_nll_optimizer import KelpieMultiClassNLLOptimizer
from link_prediction.optimization.pairwise_ranking_optimizer import KelpiePairwiseRankingOptimizer
from collections import OrderedDict
from typing import List, Dict, Tuple, Union, Optional
from config import *
from kelpie_dataset import KelpieDataset
import warnings
import threading
from xrule_dataset import XruleDataset

import scipy.stats as stats

warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim")

parser = argparse.ArgumentParser(description="Model-agnostic tool for explaining link predictions")

def get_yaml_data(yaml_file):
    # 打开yaml文件
    print("***获取yaml文件数据***")
    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    
    print("类型：", type(file_data))

    # 将字符串转化为字典或列表
    print("***转化yaml数据为字典或列表***")
    data = yaml.safe_load(file_data)
    print(data)
    print("类型：", type(data))
    return data

config = get_yaml_data('config.yaml')
'''
name="${method}${embedding_model}_${dataset}"
explain_path: input_facts/${name}.csv
model_path: stored_models/${name}.pt
output_folder: stage5/${name}
'''

parser.add_argument("--dataset",
                    type=str,
                    help="The dataset to use: FB15k, FB15k-237, WN18, WN18RR or YAGO3-10",
                    choices=["FB15k", "FB15k-237", "WN18", "WN18RR", "YAGO3-10", "MOF-3000"])

parser.add_argument("--method",
                    type=str,
                    help="The method to use: ComplEx, ConvE, TransE", 
                    choices=["ComplEx", "ConvE", "TransE", "DistMult", "TuckER", "CompGCN"])

parser.add_argument("--coverage",
                    type=int,
                    default=10,
                    help="Number of random entities to extract and convert")

parser.add_argument("--system",
                    type=str,
                    default=None,
                    choices=[None, "k1", "data_poisoning", "criage", "xrule", "kelpie", "k1+xrule", "kelpie+xrule"],
                    help="attribute to use when we want to use a system rather than the Kelpie engine")

parser.add_argument("--entities_to_convert",
                    type=str,
                    help="path of the file with the entities to convert (only used by baselines)")


parser.add_argument("--relevance_threshold",
                    type=float,
                    default=None,
                    help="The relevance acceptance threshold to use")

prefilters = [TOPOLOGY_PREFILTER, TYPE_PREFILTER, NO_PREFILTER]
parser.add_argument('--prefilter',
                    choices=prefilters,
                    default='graph-based',
                    help="Prefilter type in {} to use in pre-filtering".format(prefilters))

parser.add_argument("--prefilter_threshold",
                    type=int,
                    default=20,
                    help="The number of promising training facts to keep after prefiltering")

parser.add_argument("--run",
                    type=str,
                    default='111',
                    help="whether train, test or explain")

parser.add_argument("--output_folder",
                    type=str,
                    default='.')

parser.add_argument("--embedding_model",
                    type=str,
                    default='',
                    help="embedding model before LP model header")

parser.add_argument('--ignore_inverse', dest='ignore_inverse', default=False, action='store_true',
                    help="whether ignore inverse relation when evaluate")

parser.add_argument('--train_restrain', dest='train_restrain', default=False, action='store_true',
                    help="whether apply tail restrain when training")

parser.add_argument('--specify_relation', dest='specify_relation', default=False, action='store_true',
                    help="whether specify relation when evaluate")

parser.add_argument('--relevance_method', type=str, default='kelpie', 
                    choices=['rank', 'score', 'kelpie', 'hybrid'], help="the method to compute relevance")

parser.add_argument('--split', type=int, default=1,
                    help='explain only the specified split of testing facts')

parser.add_argument('--process', type=int, default=1,
                    help='explain only the specified split of testing facts')

parser.add_argument('--verify', default=False, action='store_true',
                    help="whether verify")

parser.add_argument('--fit_coef', default=False, action='store_true',
                    help="fit coefficient")

parser.add_argument('--perspective', type=str, default='',
                    help="verify perspective")

parser.add_argument('--top_n_explanation', type=int, default=1)

# parser.add_argument('--sort', dest='sort', default=False, action='store_true',
#                     help="whether sort the dataset")

args = parser.parse_args()
cfg = config[args.dataset][args.method]
# coef = config[args.dataset]['coef']

# coef = pd.read_csv(f'stage7/{args.method}-distribution.csv', index_col=0).loc[args.dataset].to_dict()

args.restrain_dic = config[args.dataset].get('tail_restrain', None)
# print(cfg)
args.name = f"{args.method}{args.embedding_model}_{args.dataset}"
args.already_explain_path = f"{args.output_folder}/../already_explain"
args.explain_path = f"input_facts/{args.name}.csv"
args.model_path = f"stored_models/{args.name}.pt"
os.makedirs('input_facts', exist_ok=True)
os.makedirs('stored_models', exist_ok=True)


if not args.verify:
    import shutil
    if args.process > 1:
        time.sleep((args.split-1)*0.5)
        if args.split == 1:
            # shutil.rmtree(args.already_explain_path, ignore_errors=True)
            os.makedirs(args.already_explain_path, exist_ok=True)

    if args.system == 'xrule':
        # df_lock = defaultdict(threading.Lock)
        # rv_dic = {}
        # for rv_name in coef:
        #     rv_para = coef[rv_name]
        #     # create a t distribution with df = rv_para[0], loc = rv_para[1], scale = rv_para[2]
        #     if hasattr(rv_para, '__iter__'):
        #         rv_dic[rv_name] = stats.t(df=rv_para[0], loc=rv_para[1], scale=rv_para[2])
        # print('rv distribution dic:', rv_dic)
        os.makedirs(f'{args.output_folder}/hyperpath', exist_ok=True)
        os.makedirs(f'{args.output_folder}/head', exist_ok=True)
        os.makedirs(f'{args.output_folder}/tail', exist_ok=True)
        os.makedirs(f'{args.output_folder}/log', exist_ok=True)


class CustomFormatter(logging.Formatter):
    """override logging.Formatter to use an aware datetime object"""
    def converter(self, timestamp):
        dt_object = datetime.fromtimestamp(timestamp)
        return dt_object

    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            t = dt.strftime(self.default_time_format)
            s = self.default_msec_format % (t, record.msecs)
        return s
    
global_logger = logging.getLogger(__name__)

outfile = f'{args.output_folder}/my_app.log'

file_handler = logging.FileHandler(outfile)
file_handler.setFormatter(CustomFormatter('%(asctime)s:%(levelname)s:%(message)s', '%H:%M:%S'))
# StreamHandler for logging to stdout
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(CustomFormatter('%(asctime)s:%(levelname)s:%(message)s', '%H:%M:%S'))   # .%f

global_logger.addHandler(file_handler)
# logger.addHandler(stream_handler)
global_logger.setLevel(logging.INFO)

logger_dic = {}
def get_logger():
    if MULTI_THREAD > 1:
        # pid = mp.current_process().pid
        pid = threading.get_ident()
        if pid not in logger_dic:
            logger = logging.getLogger(f'{__name__}.{pid}')
            file_handler = logging.FileHandler(f'{args.output_folder}/log/my_app_{pid}.log')
            logger.addHandler(file_handler)
            # logger.addHandler(stream_handler)
            logger.setLevel(logging.INFO)
            logger_dic[pid] = logger
        return logger_dic[pid]
    return global_logger


def log(s, warning=False):
    if warning:
        ech(s)
    else:
        get_logger().info(s)

def ech(s):
    s = '='*10 + s + '='*10
    click.echo(click.style(s, 'yellow'))
    get_logger().warning(s)


# load the dataset and its training samples
ech(f"Loading dataset {args.dataset}...")
dataset = Dataset(name=args.dataset, separator="\t", load=True, args=args)
try:
    tail_restrain = dataset.tail_restrain
except:
    tail_restrain = None
args.tail_restrain = tail_restrain

ech("Initializing LP model...")
epoches = cfg['Ep'] # // 2
hyperparameters = {
    DIMENSION: cfg['D'],
    EPOCHS: epoches,
    RETRAIN_EPOCHS: cfg['REp'] if 'REp' in cfg else epoches,
    BATCH_SIZE: cfg['B'],
    LEARNING_RATE: cfg['LR']
}
if args.method == "ConvE":
    hyperparameters = {**hyperparameters,
                    INPUT_DROPOUT: cfg['Drop']['in'],
                    FEATURE_MAP_DROPOUT: cfg['Drop']['feat'],
                    HIDDEN_DROPOUT: cfg['Drop']['h'],
                    HIDDEN_LAYER_SIZE: 9728,
                    DECAY: cfg['Decay'],
                    LABEL_SMOOTHING: 0.1}
    TargetModel = ConvE
    Optimizer = BCEOptimizer
elif args.method == "ComplEx":
    hyperparameters = {**hyperparameters,
                    INIT_SCALE: 1e-3,
                    OPTIMIZER_NAME: 'Adagrad',  # 'Adagrad', 'Adam', 'SGD'
                    DECAY_1: 0.9,
                    DECAY_2: 0.999,
                    REGULARIZER_WEIGHT: cfg['Reg'],
                    REGULARIZER_NAME: "N3"}
    TargetModel = ComplEx
    Optimizer = MultiClassNLLOptimizer
elif args.method == "TransE":
    hyperparameters = {**hyperparameters,
                    MARGIN: 5,
                    NEGATIVE_SAMPLES_RATIO: cfg['N'],
                    REGULARIZER_WEIGHT: cfg['Reg'],}
    TargetModel = TransE
    Optimizer = PairwiseRankingOptimizer
elif args.method == "DistMult":
    hyperparameters = {**hyperparameters,
                    INIT_SCALE: 1e-3,
                    OPTIMIZER_NAME: 'Adagrad',  # 'Adagrad', 'Adam', 'SGD'
                    DECAY_1: 0.9,
                    DECAY_2: 0.999,
                    REGULARIZER_WEIGHT: cfg['Reg'],
                    REGULARIZER_NAME: "N3"}
    TargetModel = DistMult
    Optimizer = MultiClassNLLOptimizer

print('LP hyperparameters:', hyperparameters)

if args.embedding_model and args.embedding_model != 'none':
    cf = config[args.dataset][args.embedding_model]
    print('embedding_model config:', cf)
    args.embedding_model = CompGCN(
        num_bases=cf['num_bases'],
        num_rel=dataset.num_relations,
        num_ent=dataset.num_entities,
        in_dim=cf['in_dim'],
        layer_size=cf['layer_size'],
        comp_fn=cf['comp_fn'],
        batchnorm=cf['batchnorm'],
        dropout=cf['dropout']
    )
else:
    args.embedding_model = None

model = TargetModel(dataset=dataset, hyperparameters=hyperparameters, init_random=True)
model.to('cuda')

if os.path.exists(args.model_path):
    ech(f'loading models from path: {args.model_path}')
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict)
else:
    ech(f'model does not exists! {args.model_path}')
    
# torch.save(model.state_dict(), f'{args.output_folder}/params.pth')
# args.state_dict = torch.load(f'{args.output_folder}/params.pth')


if isinstance(model, ComplEx):
    kelpie_optimizer_class = KelpieMultiClassNLLOptimizer
elif isinstance(model, ConvE):
    kelpie_optimizer_class = KelpieBCEOptimizer
elif isinstance(model, TransE):
    kelpie_optimizer_class = KelpiePairwiseRankingOptimizer
else:
    kelpie_optimizer_class = KelpieMultiClassNLLOptimizer
kelpie_model_class = model.kelpie_model_class()


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def rd(x):
    return np.round(x, 4)

def mean(lis):
    return rd(np.mean(lis))

def tensor_head(t):
    return [rd(x) for x in t.view(-1)[:3].detach().cpu().numpy().tolist()]

def std(lis):
    return rd(np.std(lis))

def get_removel_relevance(rank_delta, score_delta):
    if args.relevance_method == 'kelpie':
        relevance = float(rank_delta + sigmoid(score_delta))
    elif args.relevance_method == 'rank':
        relevance = rank_delta
    elif args.relevance_method == 'score':
        relevance = score_delta
    elif args.relevance_method == 'hybrid':
        relevance = np.tanh(rank_delta) + np.tanh(score_delta)
    return rd(relevance)

def extract_performances(model: Model, sample: numpy.array):
    model.eval()
    head_id, relation_id, tail_id = sample

    # print('[extract]trainable_indices', model.trainable_indices)
    # check how the model performs on the sample to explain
    all_scores = model.all_scores(numpy.array([sample])).detach().cpu().numpy()[0]
    target_entity_score = all_scores[tail_id] # todo: this only works in "head" perspective
    filter_out = model.dataset.to_filter[(head_id, relation_id)] if (head_id, relation_id) in model.dataset.to_filter else []

    if model.is_minimizer():
        all_scores[filter_out] = 1e6
        # if the target score had been filtered out, put it back
        # (this may happen in necessary mode, where we may run this method on the actual test sample;
        all_scores[tail_id] = target_entity_score
        best_entity_score = numpy.min(all_scores)
        target_entity_rank = numpy.sum(all_scores <= target_entity_score)  # we use min policy here

    else:
        all_scores[filter_out] = -1e6
        # if the target score had been filtered out, put it back
        # (this may happen in necessary mode, where we may run this method on the actual test sample;
        all_scores[tail_id] = target_entity_score
        best_entity_score = numpy.max(all_scores)
        target_entity_rank = numpy.sum(all_scores >= target_entity_score)  # we use min policy here

    return rd(target_entity_score), rd(best_entity_score), target_entity_rank

def extract_performances_on_embeddings(trainable_entities, embedding: torch.Tensor, prediction: numpy.array, grad:bool=False):
    new_model = TargetModel(dataset=dataset, hyperparameters=hyperparameters)
    new_model.load_state_dict(state_dict=args.state_dict)
    new_model = new_model.to('cuda')
    new_model.start_post_train(trainable_indices=trainable_entities, init_tensor=embedding)
    if grad:
        new_model.eval()
        return new_model.calculate_grad(prediction)
    return extract_performances(new_model, prediction)


def extract_samples_with_entity(samples, entity_id):
    return samples[numpy.where(numpy.logical_or(samples[:, 0] == entity_id, samples[:, 2] == entity_id))]

def extract_samples_with_entities(samples, entity_ids):
    return samples[numpy.where(numpy.logical_or(numpy.isin(samples[:, 0], entity_ids), numpy.isin(samples[:, 2], entity_ids)))]

def extract_training_samples_length(trainable_entities) -> np.array:
    original_train_samples = []
    for entity in trainable_entities:
        original_train_samples.extend(dataset.entity_id_2_train_samples[entity])
    # stack a list of training samples, each of them is a tuple
    return len(original_train_samples)

def mean_of_tensor_list(tensor_list):
    return torch.mean(torch.stack(tensor_list), dim=0)

def unfold(lis):
    # check if is tensor, if so, tolist
    if isinstance(lis, torch.Tensor):
        lis = lis.tolist()
    if not hasattr(lis, '__iter__'):
        return rd(lis)
    if len(lis) == 1:
        return unfold(lis[0])
    return [rd(x) for x in lis]

def get_path_entities(prediction, path):
    last_entity_on_path = prediction[0]
    path_entities = [last_entity_on_path]
    for triple in path:
        target = triple[2] if triple[0] == last_entity_on_path else triple[0]
        path_entities.append(target)
        last_entity_on_path = target
    return path_entities

def update_df(df, dic, save_path):
    # df_lock[save_path].acquire()
    for key in set(dic.keys()) - set(df.columns):
        df[key] = None
    df.loc[len(df)] = dic
    df.to_csv(f'{args.output_folder}/{save_path}', index=False)
    # df_lock[save_path].release()

def overlapping_block_division(neighbors, m):
    neighbors = list(neighbors)
    n = len(neighbors)
    k = math.ceil(math.log(n, m))
    N = m ** k
    cnt = n // m
    print(f"n: {n}, m: {m}, k: {k}, N: {N}, cnt: {cnt}")

    group_id_to_elements = {}
    element_id_to_groups = defaultdict(list)

    # fill neighbors with -1 until it has N elements
    neighbors += [-1] * (N - n)
    # create a k-dim matrix with m elements in each dimension, and fill it with the elements in neighbors
    matrix = np.array(neighbors).reshape((m,) * k)

    for i in range(k):
        # get m slices from the i-th dimension and store them in a list(group), group_id = m * i + j
        for j in range(m):
            group = matrix.take(j, axis=i).flatten()
            group_id = m * i + j
            group_id_to_elements[group_id] = [element for element in group if element != -1]
            for element in group_id_to_elements[group_id]:
                element_id_to_groups[element].append(group_id)

    return group_id_to_elements, element_id_to_groups

identifier2trainable_entities_embedding = {}


def extract_samples_with_entity(samples, entity_id):
    return samples[numpy.where(numpy.logical_or(samples[:, 0] == entity_id, samples[:, 2] == entity_id))]


def extract_samples_with_entities(samples, entity_ids):
    return samples[numpy.where(numpy.logical_or(numpy.isin(samples[:, 0], entity_ids), numpy.isin(samples[:, 2], entity_ids)))]


class Explanation:
    """
    calculate removel relevance using the same process as kelpie engine
    """
    _original_model_results = {}
    _base_pt_model_results = {}
    df = pd.DataFrame()

    _kelpie_dataset_cache_size = 20
    _kelpie_dataset_cache = OrderedDict()
    _kelpie_init_tensor_cache = {}

    def __init__(self, 
                 prediction: Tuple[Any, Any, Any],
                 samples_to_remove: List[Tuple],
                 trainable_entities: List, *args, **kwargs):
        
        log(f"Create Explanation on sample: {prediction}, trainable: {trainable_entities}, removing({len(samples_to_remove)}): {samples_to_remove}")
        self.prediction = prediction
        self.samples_to_remove = list(samples_to_remove)
        self.trainable_entities = trainable_entities
        self.original_entity_id = trainable_entities.copy()

        self.original_entity_id.sort()
        self.original_entity_id = tuple(self.original_entity_id)
        self.identifier = (prediction, self.original_entity_id)

        for entity in trainable_entities:
            assert entity in prediction

        start_time = time.time()
        # kelpie_init_tensor_size = model.dimension if not isinstance(model, TuckER) else model.entity_dimension
        # self.kelpie_init_tensor = torch.rand(len(trainable_entities), kelpie_init_tensor_size)
        self.kelpie_init_tensor = self._get_kelpie_init_tensor()
        self.kelpie_dataset = self._get_kelpie_dataset()
        self.kelpie_prediction = self.kelpie_dataset.as_kelpie_sample(prediction)
        self.kelpie_entity_id = self.kelpie_dataset.kelpie_entity_id

        base_metric = self.base_post_train()
        pt_metric = self.post_train()

        self.rank_worsening = pt_metric['rank'] - base_metric['rank']
        self.score_worsening = pt_metric['score'] - base_metric['score'] if model.is_minimizer() else base_metric['score'] - pt_metric['score']

        # note: the formulation is very different from the addition one
        # self.relevance = float(rank_worsening + self.sigmoid(score_worsening))
        self.relevance = self.score_worsening
        self.execution_time = time.time() - start_time

        def filter_keys(dic):
            return {k:v for k,v in dic.items() if k not in ['embedding', 'grad']}

        self.ret = {
            'prediction': dataset.sample_to_fact(self.prediction, True),
            'length': len(self.samples_to_remove),
            'base': filter_keys(base_metric),
            'pt': filter_keys(pt_metric),
            'rank_worsening': self.rank_worsening,
            'score_worsening': self.score_worsening,
            'relevance': self.relevance,
            'delta_2': unfold(torch.norm(base_metric['embedding'] - pt_metric['embedding'], p=2, dim=1)),
            'delta_inf': unfold(torch.norm(base_metric['embedding'] - pt_metric['embedding'], p=float('inf'), dim=1)),
            'execution_time': self.execution_time,
            **base_metric['grad']
        }

        update_df(self.df, self.ret, 'my_explaination.csv')
        log(f"MyExplanation created. {str(self.ret)}")

    
    def _get_kelpie_init_tensor(self):
        embeddings = []
        for entity in self.original_entity_id:
            if entity not in self._kelpie_init_tensor_cache:
                kelpie_init_tensor_size = model.dimension if not isinstance(model, TuckER) else model.entity_dimension
                self._kelpie_init_tensor_cache[entity] = torch.rand(1, kelpie_init_tensor_size, device='cuda') * 2 - 1   # 
            embeddings.append(self._kelpie_init_tensor_cache[entity])
        return torch.cat(embeddings, dim=0)

    
    def extract_performance_with_embedding(self, kelpie_model, calculate_grad=False):
        score, best_score, rank = extract_performances(kelpie_model, self.kelpie_prediction)
        embedding = kelpie_model.entity_embeddings[self.kelpie_entity_id]   # .detach().cpu().numpy()
        grad = kelpie_model.calculate_grad(self.kelpie_prediction) if calculate_grad else None

        return {
            'score': score,
            'best_score': best_score,
            'rank': rank,
            'embedding': embedding,
            'delta_2': unfold(torch.norm(embedding - self.kelpie_init_tensor, p=2)),
            'grad': grad
        }

    def base_post_train(self):
        if self.identifier in self._base_pt_model_results:
            return self._base_pt_model_results[self.identifier]

        kelpie_model = kelpie_model_class(model=model, dataset=self.kelpie_dataset, init_tensor=self.kelpie_init_tensor)
        kelpie_model.to('cuda')

        optimizer = kelpie_optimizer_class(model=kelpie_model,
                                                hyperparameters=hyperparameters,
                                                verbose=False)
        optimizer.epochs = hyperparameters[RETRAIN_EPOCHS]
        t = time.time()
        optimizer.train(train_samples=self.kelpie_dataset.kelpie_train_samples)
        print(f"Kelpie training time: {time.time() - t}")
        self._base_pt_model_results[self.identifier] = self.extract_performance_with_embedding(kelpie_model, calculate_grad=True)
        return self._base_pt_model_results[self.identifier]


    def post_train(self):
        kelpie_model = kelpie_model_class(model=model, dataset=self.kelpie_dataset, init_tensor=self.kelpie_init_tensor)
        kelpie_model.to('cuda')

        optimizer = kelpie_optimizer_class(model=kelpie_model,
                                                hyperparameters=hyperparameters,
                                                verbose=False)
        optimizer.epochs = hyperparameters[RETRAIN_EPOCHS]
        t = time.time()

        self.kelpie_dataset.remove_training_samples(self.samples_to_remove)
        optimizer.train(train_samples=self.kelpie_dataset.kelpie_train_samples)
        self.kelpie_dataset.undo_last_training_samples_removal()

        print(f"Kelpie training time: {time.time() - t}")
        return self.extract_performance_with_embedding(kelpie_model)
    

    def _get_kelpie_dataset(self):
        """
        Return the value of the queried key in O(1).
        Additionally, move the key to the end to show that it was recently used.

        :param original_entity_id:
        :return:
        """
        if self.original_entity_id not in self._kelpie_dataset_cache:
            self._kelpie_dataset_cache[self.original_entity_id] = XruleDataset(dataset, self.original_entity_id)
            self._kelpie_dataset_cache.move_to_end(self.original_entity_id)

            if len(self._kelpie_dataset_cache) > self._kelpie_dataset_cache_size:
                self._kelpie_dataset_cache.popitem(last=False)
        return self._kelpie_dataset_cache[self.original_entity_id]


    
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                            numpy.uint16, numpy.uint32, numpy.uint64)):
            return int(obj)
        elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32,
                              numpy.float64)):
            return float(obj)
        elif isinstance(obj, (numpy.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (numpy.bool_,)):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)


class Path:
    paths = []
    rel_df = pd.DataFrame()

    def __init__(self, prediction, path, available_samples) -> None:
        """Constructor for Path

        Args:
            prediction (_type_): _description_
            path (_type_): a list of triples connecting head and tail. The triples should be in the same order as the path
        """
        self.prediction = prediction
        self.path = path
        self.path_entities = self.get_path_entities()
        self.available_samples = available_samples
        self.build_explanations()
        self.explanations = [self.head_exp, self.tail_exp, self.path_exp]

        log(f'path relevance: {[exp.relevance for exp in self.explanations]}')

        self.save_to_local()

    @property
    def triples(self):
        return self.path

    def save_to_local(self):
        self.paths.append(self)
        self.ret = {
            'prediction': self.prediction,
            'path': self.path,
            'head_rel': self.head_exp.relevance,
            'tail_rel': self.tail_exp.relevance,
            'rel': self.path_exp.relevance,
        }
        for p in [2, float('inf')]:
            self.ret.update({
                f'delta_h_{p}': self.head_exp.ret[f'delta_{p}'],
                f'delta_t_{p}': self.tail_exp.ret[f'delta_{p}'],
                f'delta_{p}': self.path_exp.ret[f'delta_{p}'],
                f'partial_{p}': (self.head_exp.ret[f'partial_{p}'] + self.tail_exp.ret[f'partial_{p}'])/2,
                f'partial_t_{p}': (self.head_exp.ret[f'partial_t_{p}'] + self.tail_exp.ret[f'partial_t_{p}'])/2,
                f'partial_h_{p}': (self.head_exp.ret[f'partial_h_{p}'] + self.tail_exp.ret[f'partial_h_{p}'])/2,
                'head_exp': self.head_exp.ret,
                'tail_exp': self.tail_exp.ret,
                'path_exp': self.path_exp.ret,
            })
        update_df(self.rel_df, self.ret,'rel_df.csv')

        lis = [path.json() for path in self.paths]
        with open(f'{args.output_folder}/paths.json', 'w') as f:
            json.dump(lis, f, cls=NumpyEncoder, indent=4)

    def get_path_entities(self):
        return get_path_entities(self.prediction, self.path)

    def build_explanations(self):
        head, rel, tail = self.prediction
        self.head_exp = Explanation(self.prediction, [self.path[0]], [head], self.available_samples)
        self.tail_exp = Explanation(self.prediction, [self.path[-1]], [tail], self.available_samples)
        self.path_exp = Explanation(self.prediction, [self.path[0], self.path[-1]], [head, tail], self.available_samples)

    def json(self):
        return {
            'prediction': self.prediction,
            'facts': [dataset.sample_to_fact(triple, True) for triple in self.triples],
            'path': self.path,
            'head_exp': self.head_exp.ret,
            'tail_exp': self.tail_exp.ret,
            'path_exp': self.path_exp.ret,
            'relevance': [exp.relevance for exp in self.explanations],
            'ret': self.ret
        }
        

class SuperPath(Path):
    def __init__(self, prediction, path, available_samples) -> None:
        super().__init__(prediction, path, available_samples)

    def get_path_entities(self):
        return self.path

    @property
    def triples(self):
        return self.all_samples

    def build_explanations(self):
        head, rel, tail = self.prediction
        first_hop = self.path[1]
        last_hop = self.path[-2]
        print('Constructing super path:  first_hop', first_hop, 'last_hop', last_hop, 'tail', tail)
        print('last_hop samples', len(self.available_samples[last_hop]), list(self.available_samples[last_hop])[:10])
        print('tail samples', len(self.available_samples[tail]), list(self.available_samples[tail])[:10])
        
        first_hop_samples = self.available_samples[head] & self.available_samples[first_hop]
        last_hop_samples = self.available_samples[tail] & self.available_samples[last_hop]

        for sample in last_hop_samples:
            print(sample, tail, last_hop)
            assert sample[0] in [tail, last_hop] and sample[2] in [tail, last_hop]

        self.all_samples = set()
        for i in range(len(self.path)-1):
            entity = self.path[i]
            for triple in self.available_samples[entity]:
                if self.path[i+1] in [triple[0], triple[2]]:
                    self.all_samples.add(triple)

        self.head_exp = Explanation(self.prediction, first_hop_samples, [head], self.available_samples)
        self.tail_exp = Explanation(self.prediction, last_hop_samples, [tail], self.available_samples)
        self.path_exp = Explanation(self.prediction, first_hop_samples | last_hop_samples, [head, tail], self.available_samples)
        
    
class Generator:
    generator_df = pd.DataFrame()
    window_size = 5

    def __init__(self, prediction, objects, perspective, available_samples) -> None:
        self.prediction = prediction
        self.perspective = perspective
        self.objects = list(objects)
        self.windows = []
        self.upbound_dic = {}
        self.explanations = {}
        self.group_count = 0
        self.valid_count = 0
        self.last_upbound = np.inf
        self.invalid_objects = set()
        self.available_samples = available_samples
        log(f'generator initialized with {len(self.objects)} #objects, {len(self.available_samples)} #entities')

    def __len__(self):
        return len(self.upbound_dic)
    
    def valid_objects(self):
        return set(self.objects) - self.invalid_objects

    def finished(self):
        """Use a sliding window for all generators (window size=5). 
        Records all the relevance generated, 
        continue with the upbound of mean of relevance in the recent window 
        dividing the largest relevance, else finish.

        Returns:
            _type_: _description_
        """
        if self.empty():
            log('queue is empty, finish')
            return True
        if len(self.windows) < self.window_size * 5:
            log(f'window size: {len(self.windows)} < {self.window_size * 5}, continue')
            return False
        
        relu_windows = [max(0, x) for x in self.windows[-self.window_size:]]
        if max(relu_windows) <= DEFAULT_VALID_THRESHOLD:
            log('no valid relevance in the recent window, finish')
            return True
        
        if relu_windows[-1] > DEFAULT_VALID_THRESHOLD:
            log(f'last window: {relu_windows[-1]} > {DEFAULT_VALID_THRESHOLD}, continue')
            return False
        
        # if self.last_upbound > DEFAULT_VALID_THRESHOLD and len(self.windows) < self.window_size * 10:
        #     log(f'upbound: {self.last_upbound} > {DEFAULT_VALID_THRESHOLD} and window size: {len(self.windows)} < {self.window_size * 10}, continue')
        #     return False
        
        prob = mean(relu_windows) / max(self.windows)
        # generate a random number between 0 and 1
        r = random.random() * 2 - 1
        if r > prob:
            log(f'random: {r} > prob: {prob}, finish')
            return True
        
        log(f'random: {r} <= prob: {prob}, continue')
        return False
    
    def generate(self):
        pass

    def json(self):
        pass

    def empty(self):
        return len(self.upbound_dic) == 0 

    def close(self):
        update_df(self.generator_df, {
            'prediction': self.prediction,
            'perspective': self.perspective,
            'objects': self.objects,
            '#objects': len(self.objects),  # original number of objects
            '#valid': self.valid_count,    # valid number of objects
            '#explanations': len(self.explanations),    # number of calculated explanations
            '#groups': self.group_count,    # number of groups
        }, 'generator.csv')



class OneHopGenerator(Generator):
    NoSplitThresh = 30

    df = pd.DataFrame()
    def __init__(self, perspective, prediction, neighbors, available_samples=None) -> None:
        """For given perspective, generate top one hop explanation 
        (1) k-dim cross partition
        (2) select top valid upbound ph/pt, calculate and yield explanation

        Args:
            perspective (stinr): 'head' or 'tail'
            prediction (tuple): sample to explain
            neighbors (list[int]): neighbors of the perspective entity
        """
        super().__init__(prediction, neighbors, perspective, available_samples)
        self.entity = prediction[0] if perspective == 'head' else prediction[2]
        self.ele2exp = defaultdict(list)

        ech('calculate relevance for 2-hop neighbors')
        objects = list(neighbors).copy()
        for neighbor, length in neighbors.items():
            if length == 2:
                exp = self.calculate_group([neighbor])
                self.ele2exp[neighbor] = [exp]
                self.upbound_dic[neighbor] = exp.relevance + 10     # 优先2-hop
                objects.remove(neighbor)
        self.objects = objects

        if len(self.objects) <= self.NoSplitThresh:
            ech(f'no need to ODB {len(self.objects)}')
            for neighbor in self.objects:
                exp = self.calculate_group([neighbor])
                self.ele2exp[neighbor] = [exp]
                self.upbound_dic[neighbor] = exp.relevance
        else:
            m = max(math.ceil(len(self.objects) / MAX_GROUP_SIZE), 3)
            max_k = math.ceil(math.log(len(self.objects), m))
            ech(f'need to make groups (n: {len(self.objects)}, m: {m}), max_k: {max_k}')

            group_id_to_elements = defaultdict(list)
            element_id_to_groups = defaultdict(list)
            
            for k in range(max_k):
                # make a even division of objects into m groups
                new_objects = []
                for i, element in enumerate(self.objects):
                    group_id = i % m + k * m
                    group_id_to_elements[group_id].append(element)
                    element_id_to_groups[element].append(group_id)
                # replace objects with current groups

                for i in range(m):
                    group_id = i + k * m
                    group = group_id_to_elements[group_id]
                    
                    log(f'========group({group_id}): {len(group)} {group}')
                    if len(group) == 0:
                        continue
                    exp = self.calculate_group(group)
                    
                    # not a valid group
                    if group[0] in self.invalid_objects:
                        continue
                    
                    exp.group_id = group_id
                    new_objects.extend(group)
                    for element_id in group:
                        self.ele2exp[element_id].append(exp)
                    
                    # if len(self.valid_objects()) <= MAX_GROUP_SIZE:   
                    #     break
                
                if len(self.valid_objects()) <= self.NoSplitThresh:   
                    log(f'valid objects: {len(self.valid_objects())} <= {self.NoSplitThresh}, break')
                    break
                objects = new_objects
            
            for neighbor in self.valid_objects():
                self.calculate_upbound(neighbor)
            # self.neighbors.sort(key=lambda x: self.calculate_upbound(x), reverse=True)
            log(f'make groups completed, upbound ({len(self.upbound_dic)}/{len(self.objects)} valid)', True)
        
        log(f"upbound_dic: {self.upbound_dic}")
        self.valid_count = len(self.upbound_dic)
    

    def calculate_upbound(self, neighbor):
        """calculate valid upbound of a list of explanations
        P(R_hi>x) = P(xi-d_x * eta_h > x) * P(yi-d_y * eta_h > x) * ...
                = P(eta_h < (xi-x)/d_x) * P(eta_h < (yi-x)/d_y) * ...
                = Fh( (xi-x)/d_x ) * Fh( (yi-x)/d_y ) * ...
        where Fh is the CDF of eta_h.

        Returns:
            float: upbound
        """
        explanations = self.ele2exp[neighbor]
        lab = self.perspective[0]
        ret = np.inf
        for exp in explanations:
            rel_group = exp.relevance
            delta_group = exp.ret['delta_2'] * exp.ret[f'partial_{lab}_2']
            # point = (rel_group - DEFAULT_VALID_THRESHOLD) / delta_group
            # ret *= rv_dic[f'eta_{lab}'].cdf(point)
            upbound = rel_group + delta_group # * coef[f'g_{lab}']
            ret = min(ret, upbound)
        
        # if ret < DEFAULT_VALID_THRESHOLD:
        #     log(f'upbound on {neighbor}: {ret} too small, skip')
        #     self.invalid_objects.add(neighbor)
        #     return
        
        self.upbound_dic[neighbor] = ret


    def generate(self):
        """return one top explanation
        """
        log('*' * 10 + f'{self.perspective} generate (len: {len(self)})')

        if self.finished():
            log(f'No more neighbors, finished')
            return None
        
        neighbor = max(self.upbound_dic, key=self.upbound_dic.get)
        self.last_upbound = self.upbound_dic.pop(neighbor)

        ##########################################################
        # if -neg_upbound < self.valid_threshold:
        #     log(f'neighbor upbound not valid, finished')
        #     return None
        assert neighbor not in self.explanations


        log(f'generate explanation for {neighbor} ({self.perspective}, upbound: {self.last_upbound})')
        if len(self.objects) <= self.NoSplitThresh:
            exp = self.ele2exp[neighbor][0]
        else:
            exp = self.calculate_group([neighbor])
        self.explanations[neighbor] = exp
        self.windows.append(exp.relevance)

        update_df(self.df, 
            {
                'neighbor': neighbor,
                'entity': self.entity,
                'upbound': self.last_upbound,
                'perspective': self.perspective,
                'prediction': self.prediction,
                'relevance': exp.relevance,
                # neighour 在列表中的序号，如果为-1，说明是2-hop
                'order': self.objects.index(neighbor) if neighbor in self.objects else -1,
                'valid': exp.relevance >= DEFAULT_VALID_THRESHOLD,
                'triples': exp.samples_to_remove,
            }, f'onehop.csv')

        ##########################################################
        # if exp.relevance < DEFAULT_VALID_THRESHOLD:
        #     log(f'neighbor {neighbor} relevance {exp.relevance} too small, generate another time')
        #     return self.generate()
        
        with open(f'{args.output_folder}/{self.perspective}/{self.prediction}.json', 'w') as f:
            json.dump(self.json(), f, indent=4, cls=NumpyEncoder)
        
        return neighbor, exp
            
    
    def json(self):
        ret = {}
        for neighbor, exp in self.explanations.items():
            ret[neighbor] = {
                'neighbor': neighbor,
                'entity': self.entity,
                'relevance': exp.relevance,
                'order': self.objects.index(neighbor) if neighbor in self.objects else -1,
                'exp': exp.ret
            }
        ret = sorted(ret.items(), key=lambda x: x[1]['relevance'], reverse=True)
        return ret
    
    def calculate_group(self, group):
        all_samples = set()
        for neighbor in group:
            all_samples |= self.available_samples[neighbor] & self.available_samples[self.entity]
        log(f'({self.perspective}) #group: {len(group)}, #samples: {len(all_samples)}, group: {group}')
        exp = Explanation(self.prediction, list(all_samples), [self.entity], self.available_samples)

        lab = self.perspective[0]
        rel_group = exp.relevance
        # delta_group = exp.ret['delta_inf'] * exp.ret[f'partial_{lab}_inf']
        delta_group = exp.ret['delta_2'] * exp.ret[f'partial_{lab}_2']
        upbound = rel_group + delta_group # * coef[f'g_{lab}']

        if upbound < DEFAULT_VALID_THRESHOLD:
            log(f'invalid group: {group}, {upbound} too small, skip')
            for neighbor in group:
                self.invalid_objects.add(neighbor)
        else:
            self.group_count += 1
        
        return exp


class PathGenerator(Generator):
    df = pd.DataFrame()
    def __init__(self, prediction, hyperpaths, available_samples=None) -> None:
        super().__init__(prediction, hyperpaths, 'path', available_samples)
        self.head_explanations = {}
        self.tail_explanations = {}
        self.head_hyperpaths = defaultdict(set)
        self.tail_hyperpaths = defaultdict(set)

        for hyperpath in hyperpaths:
            self.head_hyperpaths[hyperpath[1]].add(hyperpath)
            self.tail_hyperpaths[hyperpath[-2]].add(hyperpath)

        self.approx_rel_dic = {}

    def renew_head(self, head, explanation):
        if head in self.head_explanations:
            return
        self.head_explanations[head] = explanation
        for hyperpath in self.head_hyperpaths[head]:
            if hyperpath[-2] in self.tail_explanations and hyperpath not in self.explanations:
                self.add_to_queue(hyperpath)


    def renew_tail(self, tail, explanation):
        if tail in self.tail_explanations:
            return
        self.tail_explanations[tail] = explanation
        for hyperpath in self.tail_hyperpaths[tail]:
            if hyperpath[1] in self.head_explanations and hyperpath not in self.explanations:
                self.add_to_queue(hyperpath)


    def add_to_queue(self, hyperpath):
        head_exp = self.head_explanations[hyperpath[1]]
        tail_exp = self.tail_explanations[hyperpath[-2]]

        partial = {
            x: (head_exp.ret[x] + tail_exp.ret[x]) / 2 for x in ['partial_t_2', 'partial_h_2', 'partial_2']
        }
        Delta_h = partial['partial_t_2'] * tail_exp.ret['delta_2']
        Delta_t = partial['partial_h_2'] * head_exp.ret['delta_2']
        Delta = partial['partial_2'] * head_exp.ret['delta_2'] * tail_exp.ret['delta_2']

        # point = (DEFAULT_VALID_THRESHOLD - head_exp.relevance) / Delta_h
        # prob *= 1 - rv_dic['xi_h'].cdf(point)

        # point = (DEFAULT_VALID_THRESHOLD - tail_exp.relevance) / Delta_t
        # prob *= 1 - rv_dic['xi_t'].cdf(point)

        # point = (DEFAULT_VALID_THRESHOLD  - head_exp.relevance - tail_exp.relevance) / Delta
        # prob *= 1 - rv_dic['xi'].cdf(point)

        head_bound = head_exp.relevance + Delta_h # * coef['k_h']
        tail_bound = tail_exp.relevance + Delta_t # * coef['k_t']

        # optional!!
        if args.method != 'TransE':
            bound = head_exp.relevance + tail_exp.relevance + Delta # * coef['k']
        else:
            bound = np.inf

        upbound = min(head_bound, tail_bound, bound)
        # if upbound < DEFAULT_VALID_THRESHOLD:
        #     log(f'upbound on {hyperpath}: {upbound} too small, skip')
        #     return
        
        # 为了增加hyperpath的多样性，如果已经有了5个相同头或尾且upbound更高的hyperpath，就不再加入，否则替换
        head_path = {}
        tail_path = {}
        for path in self.upbound_dic:
            if path[1] == hyperpath[1]:
                head_path[path] = self.upbound_dic[path]
            if path[-2] == hyperpath[-2]:
                tail_path[path] = self.upbound_dic[path]

        MAX_SAME_SIZE = 5
        if len(head_path) == MAX_SAME_SIZE:
            min_path = min(head_path, key=head_path.get)
            if head_path[min_path] > upbound:
                return
            self.explanations.pop(min_path)
            self.upbound_dic.pop(min_path)
        if len(tail_path) == MAX_SAME_SIZE:
            min_path = min(tail_path, key=tail_path.get)
            if tail_path[min_path] > upbound:
                return
            self.explanations.pop(min_path)
            self.upbound_dic.pop(min_path)

        # sort hyperpath by upbound in a descending order
        self.explanations[hyperpath] = 1   # wait to be calculated

        if len(hyperpath) - 1 == 2:
            self.upbound_dic[hyperpath] = upbound + 10 # # 优先2-hop
        else:
            self.upbound_dic[hyperpath] = upbound


    def generate(self):
        """
        return one top explanation
        You should examine whether the Generator is empty before calling this function
        """
        log('*' * 10 + f'path generate (len: {len(self)})')

        if self.finished():
            log(f'No more hyperpath, finished')
            return None
        
        hyperpath = max(self.upbound_dic, key=self.upbound_dic.get)
        self.last_upbound = self.upbound_dic.pop(hyperpath)
        # neg_upbound, hyperpath = self.queue.get()
        # self.last_upbound = -neg_upbound
        
        ##########################################################
        # even if the upbound is invalid / relevance, we still need to calculate the explanation
        # if -neg_upbound < self.valid_threshold:
        #     log(f'hyperpath upbound not valid, finished')
        #     return None
        
        head, relation, tail = self.prediction
        head_exp = self.head_explanations[hyperpath[1]]
        tail_exp = self.tail_explanations[hyperpath[-2]]

        log(f'generate explanation for {hyperpath} (upbound: {self.last_upbound})')
        
        all_samples_to_remove = set()
        for i in range(len(hyperpath) - 1):
            a = hyperpath[i]
            b = hyperpath[i+1]
            all_samples_to_remove |= self.available_samples[a] & self.available_samples[b]
        exp = Explanation(self.prediction, head_exp.samples_to_remove + tail_exp.samples_to_remove, [head, tail], self.available_samples)

        assert self.explanations[hyperpath] == 1
        self.explanations[hyperpath] = exp
        self.windows.append(exp.relevance)
         
        # concat the first of head_exp.pt_embeddings and the last of tail_exp.pt_embeddings (pt_embeddings is a tensor)
        # value tensor of shape [400] cannot be broadcast to indexing result of shape [2, 200]
        # approx_embedding = torch.stack([head_exp.pt_embeddings[0], tail_exp.pt_embeddings[-1]])
        # approx_score = extract_performances_on_embeddings([head,tail], approx_embedding, self.prediction)[0]
        # log(f'approx_score: {approx_score}, base_score: {head_exp.base_score}/{tail_exp.base_score}')
        # self.approx_rel_dic[hyperpath] = (head_exp.base_score + tail_exp.base_score)/2 - approx_score

        update_df(self.df, {
            'prediction': self.prediction,
            'super_path': hyperpath,
            'upbound': self.last_upbound,
            'relevance': exp.relevance,
            'head_rel': head_exp.relevance,
            'tail_rel': tail_exp.relevance,
            'triples': all_samples_to_remove,
            # 'approx_rel': self.approx_rel_dic[hyperpath],
        }, 'hyperpath.csv')

        ##########################################################
        # if exp.relevance < DEFAULT_VALID_THRESHOLD:
        #     log(f'hyperpath {hyperpath} relevance {exp.relevance} too small, generate another time')
        #     return self.generate()
        
        with open(f'{args.output_folder}/hyperpath/{self.prediction}.json', 'w') as f:
            json.dump(self.json(), f, indent=4, cls=NumpyEncoder)

        return hyperpath, exp

    
    def json(self):
        ret = {}
        for hyperpath, exp in self.explanations.items():
            if exp == 1:    # wait to be calculated
                continue
            head_exp = self.head_explanations[hyperpath[1]]
            tail_exp = self.tail_explanations[hyperpath[-2]]
            ret[hyperpath] = {
                'relevance': exp.relevance,
                'exp': exp.ret,
                'head_exp': head_exp.ret,
                'tail_exp': tail_exp.ret,
            }
        # sort hyperpath by rel in a descending order
        ret = sorted(ret.items(), key=lambda x: x[1]['relevance'], reverse=True)
        return ret

path_dic = {}
cnt_df = pd.DataFrame()
valid_hops_df = pd.DataFrame()
valid_exp_df = pd.DataFrame()
exp_info_df = pd.DataFrame()
hop_df = pd.DataFrame()
prefilter = args.prefilter
relevance_threshold = args.relevance_threshold

def triple2str(triple):
    return '<' +','.join(triple) + '>'

