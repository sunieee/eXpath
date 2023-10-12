from typing import Any, Tuple
import torch
from torch import nn
from dataset import Dataset
from collections import defaultdict
import numpy as np
import threading

# KEYS FOR SUPPORTED HYPERPARAMETERS (to use in hyperparameter dicts)
DIMENSION = "dimension"                         # embedding dimension, when both entity and relation embeddings have same dimension
ENTITY_DIMENSION = "entity_dimension"           # entity embedding dimension, when entity and relation embeddings have different dimensions
RELATION_DIMENSION = "relation_dimension"       # relation embedding dimension, when entity and relation embeddings have different dimensions

INPUT_DROPOUT = "input_dropout"                 # dropout rate for the input embeddings
HIDDEN_DROPOUT = "hidden_dropout"               # dropout rate after the hidden layer when there is only one hidden layer
HIDDEN_DROPOUT_1 = "hidden_dropout_1"           # dropout rate after the first hidden layer
HIDDEN_DROPOUT_2 = "hidden_dropout_2"           # dropout rate after the second hidden layer
FEATURE_MAP_DROPOUT = "feature_map_dropout"     # feature map dropout

HIDDEN_LAYER_SIZE = "hidden_layer"              # hidden layer size when there is only one hidden layer

INIT_SCALE = "init_scale"                       # downscale to operate on the initial, randomly generated embeddings
OPTIMIZER_NAME = "optimizer_name"               # name of the optimization technique: Adam, Adagrad, SGD
BATCH_SIZE = "batch_size"                       # training batch size
EPOCHS = "epochs"                               # training epochs
LEARNING_RATE = "learning_rate"                 # learning rate
DECAY = "decay"                                 #
DECAY_1 = "decay_1"                             # Adam decay 1
DECAY_2 = "decay_2"                             # Adam decay 2

MARGIN = "margin"                               # pairwise margin-based loss margin
NEGATIVE_SAMPLES_RATIO = "negative_samples"     # number of negative samples to obtain, via corruption, for each positive sample

REGULARIZER_NAME = "regularizer"                # name of the regularization technique: N3
REGULARIZER_WEIGHT = "regularizer_weight"       # weight for the regularization in the loss
LABEL_SMOOTHING = "label_smoothing"             # label smoothing value

GAMMA = "gamma"
RETRAIN_EPOCHS = "retrain_epoches"
count_dic = defaultdict(list)

def terminate_at(length, count):
    '''记录长度为length的解释有多少个'''
    count_dic[length].append(count)
    print(f'\tnumber of rules with length {length}: {count}')

def rd(x):
    return np.round(x, 4)

gpu_lock = threading.Lock()
post_train = False

class Model(nn.Module):
    """
        The Model class provides the interface that any LP model should implement.

        The responsibility of any Model implementation is to
            - store the embeddings for entities and relations
            - implement the specific scoring function for that link prediction model
            - offer methods that use that scoring function
                either to run prediction on one or multiple samples, or to run forward propagation in training

        On the contrary, training and evaluation are not performed directly by the model class,
        but require the use of an Optimizer or an Evaluator object respectively.

        All models work with entity and relation ids directly (not with their names),
        that they found from the Dataset object used to initialize the Model.

        Whenever a Model method requires samples, it accepts them in the form of 2-dimensional np.arrays,
        where each row corresponds to a sample and contains the integer ids of its head, relation and tail.
    """

    def __init__(self, dataset: Dataset):
        nn.Module.__init__(self)
        self.dataset = dataset

    def is_minimizer(self):
        """
            This method specifies whether this model aims at minimizing of maximizing scores.
            :return: True if in this model low scores are better than high scores; False otherwise.
        """
        pass

    def score(self, samples: np.array) -> np.array:
        """
            This method computes and returns the plausibility scores for a collection of samples.

            :param samples: a np array containing all the samples to score
            :return: the computed scores, as a np array
        """
        pass

    # override
    def all_scores(self, samples: np.array):
        """
            This method computes, For each of the passed samples, the score for all possible tail entities.
            :param samples: a 2-dimensional np array containing the samples to score, one per row
            :return: a 2-dimensional np array that, for each sample, contains a row for each passed sample
                     and a column for each possible tail
        """

        pass

    def score_embeddings(self, 
                         head_embeddings: torch.Tensor,
                         rel_embeddings: torch.Tensor,
                         tail_embeddings: torch.Tensor):
        """We can use this method to calculate the score of a triple given its embeddings.
        And gradients can be calculated based on the embeddings.

        Args:
            head_embeddings (torch.Tensor): _description_
            rel_embeddings (torch.Tensor): _description_
            tail_embeddings (torch.Tensor): _description_
        """

        pass

    def calculate_grad(self, prediction: Tuple[Any, Any, Any]):
        """ 
        f = f(h,r,t)
        partial = \frac{\partial^2 f}{\partial h \partial t}
        partial_h = \frac{\partial f}{\partial h}
        partial_t = \frac{\partial f}{\partial t}

        Args:
            prediction (Tuple[Any, Any, Any]): _description_
        """
        assert len(prediction) == 3
        samples = np.array([prediction])

        head_embeddings = self.entity_embeddings[samples[:, 0]]
        relation_embeddings = self.relation_embeddings[samples[:, 1]]
        tail_embeddings = self.entity_embeddings[samples[:, 2]]
        
        output = self.score_embeddings(head_embeddings, relation_embeddings, tail_embeddings)
        grad = torch.autograd.grad(outputs=output, inputs=(head_embeddings, tail_embeddings), grad_outputs=torch.ones_like(output), create_graph=True,  retain_graph=True)
        dh_dt_grad = torch.autograd.grad(outputs=grad[0], inputs=tail_embeddings, grad_outputs=torch.ones_like(grad[0]), create_graph=True,  retain_graph=True)
        dt_dh_grad = torch.autograd.grad(outputs=grad[1], inputs=head_embeddings, grad_outputs=torch.ones_like(grad[1]), create_graph=True,  retain_graph=True)

        all_ret = {}
        for p in [2, float('inf')]:
            ret = {
                f'partial_{p}': (torch.norm(dh_dt_grad[0], p=p) + torch.norm(dt_dh_grad[0], p=p))/2,
                f'partial_t_{p}': torch.norm(grad[1], p=p),
                f'partial_h_{p}': torch.norm(grad[0], p=p)
            }
            all_ret.update(ret)

        # detach and move all the values tensor to cpu
        for k, v in all_ret.items():
            all_ret[k] = rd(v.cpu().detach().numpy().item())

        return all_ret


    def forward(self, samples: np.array):
        """
            This method performs forward propagation for a collection of samples.
            This method is only used in training, when an Optimizer calls it passing the current batch of samples.

            This method returns all the items needed by the Optimizer to perform gradient descent in this training step.
            Such items heavily depend on the specific Model implementation;
            they usually include the scores for the samples (in a form usable by the ML framework, e.g. torch.Tensors)
            but may also include other stuff (e.g. the involved embeddings themselves, that the Optimizer
            may use to compute regularization factors)

            :param samples: a np array containing all the samples to perform forward propagation on
        """
        pass

    def predict_samples(self, samples: np.array) -> Tuple[Any, Any, Any]:
        """
            This method performs prediction on a collection of samples, and returns the corresponding
            scores, ranks and prediction lists.

            All the passed samples must be DIRECT samples in the original dataset.
            (if the Model supports inverse samples as well,
            it should invert the passed samples while running this method)

            :param samples: the direct samples to predict, in np array format
            :return: this method returns three lists:
                        - the list of scores for the passed samples,
                                    OR IF THE MODEL SUPPORTS INVERSE FACTS
                            the list of couples <direct sample score, inverse sample score>,
                            where the i-th score refers to the i-th sample in the input samples.

                        - the list of couples (head rank, tail rank)
                            where the i-th couple refers to the i-th sample in the input samples.

                        - the list of couples (head_predictions, tail_predictions)
                            where the i-th couple refers to the i-th sample in the input samples.
                            The head_predictions and tail_predictions for each sample
                            are np arrays containing all the predicted heads and tails respectively for that sample.
        """
        pass

    def predict_sample(self, sample: np.array) -> Tuple[Any, Any, Any]:
        """
            This method performs prediction on one (direct) sample, and returns the corresponding
            score, ranks and prediction lists.

            :param sample: the sample to predict, as a np array.
            :return: this method returns 3 items:
                    - the sample score
                             OR IF THE MODEL SUPPORTS INVERSE FACTS
                      a couple containing the scores of the sample and of its inverse

                    - a couple containing the head rank and the tail rank

                    - a couple containing the head_predictions and tail_predictions np arrays;
                        > head_predictions contains all entities predicted as heads, sorted by decreasing plausibility
                        [NB: the target head will be in this np array in position head_rank-1]
                        > tail_predictions contains all entities predicted as tails, sorted by decreasing plausibility
                        [NB: the target tail will be in this np array in position tail_rank-1]
        """

        assert sample[1] < self.dataset.num_direct_relations

        scores, ranks, predictions = self.predict_samples(np.array([sample]))
        return scores[0], ranks[0], predictions[0]

    def kelpie_model_class(self):
        """
            This method provides the KelpieModel implementation class corresponding to this specific Model class.
            E.g. ComplEx.kelpie_model_class() -> KelpieComplEx.__class__

            When called on a KelpieModel subclass, it raises an exception.

            :return: The KelpieModel class that corresponds to the Model Class running this method
            :raise: an Exception if called on a KelpieModel subclass
        """
        pass


class KelpieModel(Model):
    """
        The KelpieModel class provides the interface that any post-trainable LP model should implement.

        The main functions of KelpieModel are thus identical to Model
        (which is why KelpieModel extends Model).

        In addition to that, a KelpieModels also
    """

    # override
    def predict_samples(self,
                        samples: np.array,
                        original_mode: bool = False):
        """
        This method interface overrides the superclass method by adding the option to run predictions in original_mode,
        which means ignoring in any circumstances the additional "fake" kelpie entity.

        :param samples: the DIRECT samples. They will be inverted to perform head prediction
        :param original_mode: a boolean flag specifying whether to work in original_mode or to use the kelpie entity

        :return: a np array containing
        """
        pass


    # Override
    def predict_sample(self,
                       sample: np.array,
                       original_mode: bool = False):
        """
        This method interface overrides the superclass method by adding the option to run predictions in original_mode,
        which means ignoring in any circumstances the additional "fake" kelpie entity.

        :param sample: the DIRECT sample. It will be inverted to perform head prediction
        :param original_mode: a boolean flag specifying whether to work in original_mode or to use the kelpie entity

        :return:
        """

    # this is necessary
    def update_embeddings(self):
        with torch.no_grad():
            self.entity_embeddings[self.kelpie_entity_id] = self.kelpie_entity_embedding

    #override
    def train(self, mode=True):
        """
        This method overrides the traditional train() implementation of torch.nn.Module,
        in which a.train() sets all children of a to train mode.

        In KelpieModels, in post-training any layers, including BatchNorm1d or BatchNorm2d,
        must NOT be put in train mode, because even in post-training they MUST remain constant.

        So this method overrides the traditional train() by skipping any layer children
        :param mode:
        :return:
        """
        self.training = mode
        for module in self.children():
            if not (isinstance(module, Model) or
                    isinstance(module, torch.nn.BatchNorm1d) or
                    isinstance(module, torch.nn.BatchNorm2d) or
                    isinstance(module, torch.nn.Linear) or
                    isinstance(module, torch.nn.Conv2d)):
                module.train(mode)
        return self


    def kelpie_model_class(self):
        raise Exception(self.__class__.name  + " is a KelpieModel.")