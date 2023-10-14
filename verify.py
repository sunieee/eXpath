from utils import *
from config import *

import copy
import warnings
warnings.filterwarnings('ignore')


if args.process > 1:
    # list all the folders in the output folder, grep all csv files in these folders
    # and merge them into one csv file
    ech(f'merge all the *.csv & output.txt in {args.output_folder}')
    all_csv_files = []
    for folder in os.listdir(args.output_folder):
        if not os.path.isdir(os.path.join(args.output_folder, folder)):
            continue
        for file in os.listdir(os.path.join(args.output_folder, folder)):
            if file.endswith('.csv') or file == 'output.txt':
                all_csv_files.append(os.path.join(args.output_folder, folder, file))
    ech(f'len(all_csv_files): {len(all_csv_files)}')
    # df = pd.concat([pd.read_csv(x) for x in all_csv_files])

     # combine all files in the list
    file_dic = defaultdict(list)
    for filename in all_csv_files:
        file_dic[filename.split('/')[-1]].append(filename)

    for filename, files in file_dic.items():
        if filename == 'output.txt':
            # read all txt files and merge them into one txt file
            with open(f"{args.output_folder}/{filename}", 'w') as outfile:
                for fname in files:
                    with open(fname) as infile:
                        outfile.write(infile.read())
        else:
            combined_csv = pd.concat([pd.read_csv(f) for f in files])
            combined_csv.to_csv(f"{args.output_folder}/{filename}", index=False, encoding='utf-8-sig')


'''
system:
data_poisoning
k1
kelpie
criage
xrule
k1+xrule
kelpie+xrule


dataset=YAGO3-10
method=ConvE
output_folder=test/${method}_${dataset}
CUDA_VISIBLE_DEVICES=2 python verify.py --output_folder $output_folder --dataset $dataset --method $method --process 10 --verify
'''

system = args.system
baseline = system.split('+')[0]

ech('verify all the explanations to see deterorition')

if system.count('+') or system.count('xrule'):
    xrule_folder = args.output_folder.replace(baseline, 'xrule')
    df = pd.read_csv(f'{xrule_folder}/onehop.csv')
    path_df = pd.read_csv(f'{xrule_folder}/hyperpath.csv')
    all_df = {
        'head': df[df['perspective'] == 'head'],
        'tail': df[df['perspective'] == 'tail'],
        'path': path_df,
        'all': pd.concat([df, path_df])
    }
    setting = 'path'
    ech(f'processing {setting} setting')
    path_df = all_df[setting]
else:
    path_df = None

if system.count('+') or system.count('xrule') == 0:
    output_file = f"{args.output_folder}/output.txt"
    with open(output_file, "r") as f:
        input_lines = f.readlines()
else:
    input_lines = []


out_df = pd.DataFrame()
suffix = f'(top{args.top_n_explanation})'
out_file = f"experiments/{system}{suffix}_necessary_{args.method.lower()}_{args.dataset.lower().replace('-', '')}.csv"
out_tmp = f'output_end_to_end_{system}{suffix}.csv'
ech(f'output to {out_file} and {args.output_folder}/{out_tmp}')

original_model = TargetModel(dataset=dataset,
                hyperparameters=hyperparameters,
                init_random=True) # type: ConvE
original_model.to('cuda')
original_model.load_state_dict(torch.load(args.model_path))
original_model.eval()

samples_to_explain = set()
perspective = "head"    # for all samples the perspective was head for simplicity
sample_2_top_n_rules = defaultdict(list)


logit = lambda x: -math.log((1 / (x + 1e-8)) - 1)
get_score_worsening = lambda x: logit(math.modf(x+100000)[0])


def extract_first_hops(rule_relevance_input, index, first_hop2paths):  # rule_relevance_inputs[i]
    best_rule, best_rule_relevance_str = rule_relevance_input.split(":")
    best_rule_relevance = float(best_rule_relevance_str.strip('[]'))
    best_rule_bits = best_rule.split(";")
    j = 0
    best_rule_facts = []
    while j < len(best_rule_bits):
        cur_head_name = best_rule_bits[j]
        cur_rel_name = best_rule_bits[j + 1]
        cur_tail_name = best_rule_bits[j + 2]

        best_rule_facts.append((cur_head_name, cur_rel_name, cur_tail_name))
        j += 3

    best_rule_samples = [dataset.fact_to_sample(x) for x in best_rule_facts]
    print(f'top {index} rules:', best_rule_facts, 'relevances:', best_rule_relevance)

    first_hops = set()
    for first_hop in best_rule_samples:
        if tuple(first_hop) in first_hop2paths:
            first_hops.add(tuple(first_hop))

    if len(first_hops) == 0:
        print(f"\tNo path found for any sample in {best_rule_samples}, ignore! ")
        return None, None
    
    print('\tfirst_hops:', first_hops)
    return first_hops, best_rule_relevance


def compute_degree(path):
    """
    计算路径中节点的度数和
    参数:  path (list of str): 路径。
    """
    degree = 0
    for triple in path:
        degree += len(dataset.entity_id_2_train_samples[triple[0]])
        degree += len(dataset.entity_id_2_train_samples[triple[2]])
    return degree


def random_shortest_path(paths):
    """
    从给定的路径列表中随机选择一个最短长度的路径。
    参数:  paths (list of str): 路径列表。
    返回: str: 最短长度的随机路径。
    """
    if not paths:
        return None  # 如果路径列表为空，返回None
    # 找到最小长度
    min_length = min(len(path) for path in paths)
    # 从列表中筛选出最小长度的所有元素
    shortest_paths = [path for path in paths if len(path) == min_length]

    # 在所有最短路径中选择节点度数最小的路径
    shortest_path = shortest_paths[0]
    min_degree = compute_degree(shortest_path)
    for path in shortest_paths:
        degree = compute_degree(path)
        if degree < min_degree:
            min_degree = degree
            shortest_path = path
    return shortest_path

###############################################################
# extract sample from kelpie
###############################################################
all_first_hop2paths = {}
ix = 0
while ix <= len(input_lines) -3:
    fact_line = input_lines[ix].strip()
    rules_line = input_lines[ix + 1].strip()
    empty_line = input_lines[ix + 2].strip()
    ix += 3

    # print('length of lines: ', len(fact_line), len(rules_line), len(empty_line))
    assert empty_line == ""
    assert fact_line != ""
    assert rules_line != ""

    # sample to explain
    fact = tuple(fact_line.split(";"))
    print('[fact:', dataset.fact_to_sample(fact), fact, ']')
    # print(f"fact: {fact}")
    sample = (dataset.entity_name_2_id[fact[0]], dataset.relation_name_2_id[fact[1]], dataset.entity_name_2_id[fact[2]])
    samples_to_explain.add(sample)

    all_paths = dataset.find_all_path_within_k_hop(sample[0], sample[2], 3)
    first_hop2paths = defaultdict(list)
    for path in all_paths:
        # 过滤掉长度为1的路径，这种路径是一种简单的推导，不能作为解释
        if len(path) > 1:
            first_hop2paths[tuple(path[0])].append(path)
    all_first_hop2paths[sample] = first_hop2paths

    if rules_line == "":
        continue 
    rule_relevance_inputs = rules_line.split(",")
    
    index = 0
    for rule_relevance_input in rule_relevance_inputs:
        first_hops, best_rule_relevance = extract_first_hops(rule_relevance_input, index, first_hop2paths)
        if first_hops is not None:
            sample_2_top_n_rules[sample].append((first_hops, 
                        get_score_worsening(best_rule_relevance), 'baseline'))
            index += 1

    # 即使是对于kelpie，也按照score，而不是rank排序
    sample_2_top_n_rules[sample] = sorted(sample_2_top_n_rules[sample], key=lambda x: x[1], reverse=True)


###############################################################
# extract sample from xrule
###############################################################
if path_df is not None:
    predictions = [eval(x) for x in set(path_df['prediction'])]
    top_n_count = 0
    for prediction in predictions:
        samples_to_explain.add(prediction)
        pre_df = path_df[path_df['prediction'] == str(prediction)]
        pre_df.sort_values(by=['relevance'], ascending=False, inplace=True)
        
        for i in range(min(len(pre_df), args.top_n_explanation)):
            sample_2_top_n_rules[prediction].append((list(eval(pre_df.iloc[i]['triples'])),
                                                        pre_df.iloc[i]['relevance'],
                                                        'xrule'))

        sample_2_top_n_rules[prediction] = sorted(sample_2_top_n_rules[prediction], key=lambda x: x[1], reverse=True)
        for i in range(min(len(sample_2_top_n_rules[prediction]), args.top_n_explanation)):
            top_n_count += (sample_2_top_n_rules[prediction][i][2] == 'xrule')

    top_n_ratio = top_n_count / len(predictions) / args.top_n_explanation
    print(f"============top_n_ratio: {top_n_ratio}============")


samples_to_explain = list(samples_to_explain)
print('samples_to_explain: ', samples_to_explain)
print('sample_2_top_n_rules: ', sample_2_top_n_rules)
samples_to_remove = []  # the samples to remove from the training set before retraining
sample2exp = {}

for sample in samples_to_explain:
    cur_samples_to_remove = []

    best_rule_samples = sample_2_top_n_rules[sample]
    all_first_hops = set()

    for ix, rule in enumerate(best_rule_samples[:args.top_n_explanation]):
        if rule[2] == 'baseline':
            all_first_hops.update(rule[0])
        else:
            if ix > math.ceil(args.top_n_explanation / 2) and rule[1] <= DEFAULT_VALID_THRESHOLD:
                break
            cur_samples_to_remove += rule[0]

    for first_hop in all_first_hops:
        paths = all_first_hop2paths[sample][tuple(first_hop)]
        cur_samples_to_remove += random_shortest_path(paths)

    # if system.count('+') or system.count('xrule'):
    #     for rule in best_rule_samples:
    #         if rule[2] == baseline:
    #             cur_samples_to_remove += rule[0]
    #             break
    cur_samples_to_remove = list(set(cur_samples_to_remove))

    samples_to_remove += cur_samples_to_remove
    sample2exp[sample] = cur_samples_to_remove

# samples_to_remove += best_rule_samples
samples_to_remove = list(set(samples_to_remove))


print(f"Removing samples: {len(samples_to_remove)}", samples_to_remove)
print("Removing samples: ")
for sample in samples_to_remove:
    print("\t" + dataset.printable_sample(sample))

new_dataset = copy.deepcopy(dataset)
# remove the samples_to_remove from training samples of new_dataset (and update new_dataset.to_filter accordingly)
new_dataset.remove_training_samples(numpy.array(samples_to_remove))

if len(samples_to_explain) <=1:
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
    samples_to_explain = testing_samples

# obtain tail ranks and scores of the original model for all samples_to_explain
original_scores, original_ranks, original_predictions = original_model.predict_samples(numpy.array(samples_to_explain))


# Define a function to train and evaluate a model
def train_and_evaluate_model(new_dataset):
    # Initialize the model
    if multi_thread:
        this_dataset = copy.deepcopy(new_dataset)
    else:
        this_dataset = new_dataset
    new_model = TargetModel(dataset=this_dataset, hyperparameters=hyperparameters, init_random=True)
    new_model.to('cuda')
    new_optimizer = Optimizer(model=new_model, hyperparameters=hyperparameters)
    
    # Train the model
    new_optimizer.train(train_samples=this_dataset.train_samples)
    
    # Evaluate the model
    new_model.eval()
    new_scores, new_ranks, _ = new_model.predict_samples(np.array(samples_to_explain))
    
    # Return the results
    return new_scores, new_ranks

# Create and start threads for each model
threads = []
results = []

if multi_thread:
    for _ in range(VERIFY_TIMES):
        thread = threading.Thread(target=lambda: results.append(train_and_evaluate_model(new_dataset)))
        threads.append(thread)
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()
else:
    for _ in range(VERIFY_TIMES):
        results.append(train_and_evaluate_model(new_dataset))

multiple_scores = [scores for scores, _ in results]
multiple_ranks = [ranks for _, ranks in results]


for i in range(len(samples_to_explain)):
    cur_sample = samples_to_explain[i]
    original_direct_score = original_scores[i][0]
    original_tail_rank = original_ranks[i][1]

    new_scores = [new_scores[i][0] for new_scores in multiple_scores]
    new_ranks = [new_ranks[i][1] for new_ranks in multiple_ranks]

    print("<" + ", ".join([dataset.entity_id_2_name[cur_sample[0]],
                            dataset.relation_id_2_name[cur_sample[1]],
                            dataset.entity_id_2_name[cur_sample[2]]]) + ">")
    print("\tDirect score: from " + str(original_direct_score) + " to " + str(new_scores))
    print("\tTail rank: from " + str(original_tail_rank) + " to " + str(new_ranks))
    print()

    update_df(out_df, {
            'prediction': cur_sample,
            'facts': sample_2_top_n_rules[cur_sample],
            'exp': sample2exp[cur_sample],
            'exp_length': len(sample2exp[cur_sample]),
            'original_score': original_direct_score,
            'original_rank': original_tail_rank,
            'new_scores': new_scores,
            'new_ranks': new_ranks,
            'new_score': numpy.mean(new_scores),
            'new_rank': numpy.mean(new_ranks)
        }, out_tmp)
    
out_df.to_csv(out_file, index=False)