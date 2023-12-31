import copy
from typing import Tuple, Any
import numpy as np
import torch
from torch.nn import Parameter
from torch.nn.init import xavier_normal_

from dataset import Dataset
from kelpie_dataset import KelpieDataset
from link_prediction.models.model import *
from link_prediction.models.model import Dataset
from typing import List

class ConvE(Model):
    """
        The ConvE class provides a Model implementation in PyTorch for the ConvE system.

        In training or evaluation, ConvE class requires samples to be passed as 2-dimensional np.arrays.
        Each row corresponds to a sample and contains the integer ids of its head, relation and tail.
        Only *direct* samples should be passed to the model.

        TODO: add documentation about inverse facts and relations
        TODO: explain that the input must always be direct facts only
    """


    def __init__(self,
                 dataset: Dataset,
                 hyperparameters: dict,
                 init_random = True):
        """
            Constructor for ConvE model.

            :param dataset: the Dataset on which to train and evaluate the model
            :param hyperparameters: hyperparameters dictionary.
                                    Must contain at least todo: add
        """

        # note: the init_random parameter is important because when initializing a KelpieModel,
        #       self.entity_embeddings and self.relation_embeddings must not be initialized as Parameters!

        # initialize this object both as a Model and as a nn.Module
        Model.__init__(self, dataset)

        self.args = dataset.args
        self.embedding_model = self.args.embedding_model
        self.kelpie_entity_embedding = None
        self.name = "ConvE"
        self.dataset = dataset
        self.num_entities = dataset.num_entities                                # number of entities in dataset
        self.num_relations = dataset.num_relations                              # number of relations in dataset

        self.hyperparameters = hyperparameters
        self.batch_size = hyperparameters[BATCH_SIZE]
        self.dimension = hyperparameters[DIMENSION]                             # embedding dimension
        self.input_dropout_rate = hyperparameters[INPUT_DROPOUT]               # rate of the dropout to apply to the input
        self.feature_map_dropout_rate = hyperparameters[FEATURE_MAP_DROPOUT]    # rate of the dropout to apply to the 2D feature map
        self.hidden_dropout_rate = hyperparameters[HIDDEN_DROPOUT]              # rate of the dropout to apply after the first hidden layer
        self.hidden_layer_size = hyperparameters[HIDDEN_LAYER_SIZE]             # size of the hidden layer

        # note: before passing them to the convolutional layer, ConvE will reshape the head and relation embedding
        self.embedding_width = 20                                       # width of the 2D matrix to obtain with the reshaping
        self.embedding_height = self.dimension // self.embedding_width  # corresponding height based on the embedding dimension

        # convolutional layer parameters
        self.conv_kernel_shape = (3, 3)     # convolutional kernel shape
        self.num_conv_filters = 32          # number of convolutional filters

        self.input_dropout = torch.nn.Dropout(self.input_dropout_rate).cuda()
        self.feature_map_dropout = torch.nn.Dropout2d(self.feature_map_dropout_rate).cuda()
        self.hidden_dropout = torch.nn.Dropout(self.hidden_dropout_rate).cuda()
        self.batch_norm_1 = torch.nn.BatchNorm2d(1).cuda()
        self.batch_norm_2 = torch.nn.BatchNorm2d(self.num_conv_filters).cuda()
        self.batch_norm_3 = torch.nn.BatchNorm1d(self.dimension).cuda()
        self.convolutional_layer = torch.nn.Conv2d(1, self.num_conv_filters, self.conv_kernel_shape, 1, 0, bias=True).cuda()
        #self.register_parameter('b', Parameter(torch.zeros(self.num_entities)))     # ?
        self.hidden_layer = torch.nn.Linear(self.hidden_layer_size, self.dimension).cuda()
        self.end_post_train()

        # create the embeddings for entities and relations as Parameters.
        # We do not use the torch.Embeddings module here in order to keep the code uniform to the post-training model,
        # (on which torch.Embeddings can not be used as they do not allow the post-training mechanism).
        # We have verified that this does not affect performances in any way.
        # Each entity embedding and relation embedding is instantiated with size dimension
        # and initialized with Xavier Glorot's normalization
        if init_random and not self.embedding_model:
            self.entity_embeddings = Parameter(torch.rand(self.num_entities, self.dimension).cuda(), requires_grad=True)
            self.relation_embeddings = Parameter(torch.rand(self.num_relations, self.dimension).cuda(), requires_grad=True)
            xavier_normal_(self.entity_embeddings)
            xavier_normal_(self.relation_embeddings)

    def is_minimizer(self):
        """
        This method specifies whether this model aims at minimizing of maximizing scores .
        :return: True if in this model low scores are better than high scores; False otherwise.
        """
        return False

    def forward(self, samples: np.array):
        """
            Perform forward propagation on the passed samples
            :param samples: a 2-dimensional np array containing the samples to use in forward propagation, one per row
            :return: a tuple containing
                        - the scores for each passed sample with all possible tails
                        - a partial result to use in regularization
        """
        return self.all_scores(samples)

    def score(self, samples: np.array) -> np.array:
        """
            Compute scores for the passed samples
            :param samples: a 2-dimensional np array containing the samples to score, one per row
            :return: a np array containing the scores of the passed samples
        """
        # compute scores for each possible tail
        all_scores = self.all_scores(samples)

        # extract from all_scores the specific scores for the initial samples
        samples_scores = []
        for i, (head_index, relation_index, tail_index) in enumerate(samples):
            samples_scores.append(all_scores[i][tail_index])

        return np.array(samples_scores)

    def score_embeddings(self,
                         head_embeddings: torch.Tensor,
                         rel_embeddings: torch.Tensor,
                         tail_embeddings: torch.Tensor):

        head_embeddings = head_embeddings.view(-1, 1, self.embedding_width, self.embedding_height)
        rel_embeddings = rel_embeddings.view(-1, 1, self.embedding_width, self.embedding_height)

        stacked_inputs = torch.cat([head_embeddings, rel_embeddings], 2)
        stacked_inputs = self.batch_norm_1(stacked_inputs)
        stacked_inputs = self.input_dropout(stacked_inputs)

        feature_map = self.convolutional_layer(stacked_inputs)
        feature_map = self.batch_norm_2(feature_map)
        feature_map = torch.relu(feature_map)
        feature_map = self.feature_map_dropout(feature_map)
        feature_map = feature_map.view(feature_map.shape[0], -1)

        x = self.hidden_layer(feature_map)
        x = self.hidden_dropout(x)
        x = self.batch_norm_3(x)
        x = torch.relu(x)
        scores = torch.mm(x, tail_embeddings.transpose(1, 0))

        #x += self.b.expand_as(x)
        scores = torch.sigmoid(scores)
        output_scores = torch.diagonal(scores)

        return output_scores


    def criage_first_step(self, samples: np.array):
        entity_embeddings, relation_embeddings = self.get_embedding()

        head_embeddings = entity_embeddings[samples[:, 0]]
        head_embeddings = head_embeddings.view(-1, 1, self.embedding_width, self.embedding_height)

        # list of relation embeddings for the relations of the heads
        relation_embeddings = relation_embeddings[samples[:, 1]]
        relation_embeddings = relation_embeddings.view(-1, 1, self.embedding_width, self.embedding_height)

        # tail_embeddings = entity_embeddings[samples[:, 2]].view(-1, 1, self.embedding_width, self.embedding_height)

        stacked_inputs = torch.cat([head_embeddings, relation_embeddings], 2)
        stacked_inputs = self.batch_norm_1(stacked_inputs)
        stacked_inputs = self.input_dropout(stacked_inputs)

        feature_map = self.convolutional_layer(stacked_inputs)
        feature_map = self.batch_norm_2(feature_map)
        feature_map = torch.relu(feature_map)
        feature_map = self.feature_map_dropout(feature_map)
        feature_map = feature_map.view(feature_map.shape[0], -1)

        x = self.hidden_layer(feature_map)
        x = self.hidden_dropout(x)
        x = self.batch_norm_3(x)
        x = torch.relu(x)

        return x

    def criage_last_step(self,
                         x: torch.Tensor,
                         tail_embeddings: torch.Tensor):
        scores = torch.mm(x, tail_embeddings.transpose(1, 0))

        #x += self.b.expand_as(x)
        scores = torch.sigmoid(scores)
        output_scores = torch.diagonal(scores)
        return output_scores
    
    def start_post_train(self, trainable_indices: List[int], init_tensor: torch.Tensor = None):
        # get ready for post-training
        self.frozen_entity_embeddings = self.entity_embeddings.clone().detach()
        self.trainable_indices = trainable_indices
        self.frozen_indices = list(set(range(self.entity_embeddings.shape[0])) - set(trainable_indices))
        if init_tensor is None:
            # init_tensor = torch.rand_like(self.entity_embeddings[self.trainable_indices]) - 0.5
            init_tensor = self.entity_embeddings[self.trainable_indices]

        # THIS IS IMPORTANT: the init_tensor should be cloned, not directly assigned to init_embeddings.
        # Otherwise, the gradients will be propagated to kelpie_init_tensor!!! The result will be better and better

        # init_embeddings = self.entity_embeddings[self.trainable_indices] + init_tensor.clone()
        init_embeddings = init_tensor.clone()
        self.trainable_entity_embeddings = nn.Parameter(init_embeddings, requires_grad=True)
        # self.update_embeddings()

    def calculate_g(self, head_embeddings, relation_embeddings):
        head_embeddings = head_embeddings.view(-1, 1, self.embedding_width, self.embedding_height)
        relation_embeddings = relation_embeddings.view(-1, 1, self.embedding_width, self.embedding_height)

        stacked_inputs = torch.cat([head_embeddings, relation_embeddings], 2)
        stacked_inputs = self.batch_norm_1(stacked_inputs)
        stacked_inputs = self.input_dropout(stacked_inputs)

        feature_map = self.convolutional_layer(stacked_inputs)
        feature_map = self.batch_norm_2(feature_map)
        feature_map = torch.relu(feature_map)
        feature_map = self.feature_map_dropout(feature_map)
        feature_map = feature_map.view(feature_map.shape[0], -1)

        x = self.hidden_layer(feature_map)
        x = self.hidden_dropout(x)
        x = self.batch_norm_3(x)
        return torch.relu(x)


    def calculate_grad(self, sample_to_explain: np.array):
        assert len(sample_to_explain) == 3
        samples = np.array([sample_to_explain])

        if self.trainable_indices is not None:
            entity_embeddings = torch.zeros_like(self.entity_embeddings, device='cuda')
            with torch.no_grad():
                entity_embeddings[self.frozen_indices] = self.frozen_entity_embeddings[self.frozen_indices]
            entity_embeddings[self.trainable_indices] = self.trainable_entity_embeddings
        else:
            entity_embeddings  = self.entity_embeddings

        head_embeddings = entity_embeddings[samples[:, 0]]
        relation_embeddings = self.relation_embeddings[samples[:, 1]]
        tail_embeddings = entity_embeddings[samples[:, 2]]
        
        g = self.calculate_g(head_embeddings, relation_embeddings)
        output = torch.mm(g, tail_embeddings.transpose(1, 0))

        grad_g = torch.autograd.grad(outputs=g, inputs=head_embeddings, grad_outputs=torch.ones_like(g), create_graph=True,  retain_graph=True)

        grad = torch.autograd.grad(outputs=output, inputs=(head_embeddings, tail_embeddings), grad_outputs=torch.ones_like(output), create_graph=True,  retain_graph=True)
        dh_dt_grad = torch.autograd.grad(outputs=grad[0], inputs=tail_embeddings, grad_outputs=torch.ones_like(grad[0]), create_graph=True,  retain_graph=True)
        dt_dh_grad = torch.autograd.grad(outputs=grad[1], inputs=head_embeddings, grad_outputs=torch.ones_like(grad[1]), create_graph=True,  retain_graph=True)

        all_ret = {}
        for p in [2, float('inf')]:
            ret = {
                f'partial_{p}': torch.norm(grad_g[0], p=p),
                f'partial_t_{p}': torch.norm(g, p=p),
                f'partial_h_{p}': torch.norm(grad_g[0]) * torch.norm(tail_embeddings, p=p),
                # f'dh_dt_grad_{p}': torch.norm(dh_dt_grad[0], p=p),
                # f'dt_dh_grad_{p}': torch.norm(dt_dh_grad[0], p=p),
                # f'dh_grad_{p}': torch.norm(grad[0], p=p),
                # f'dt_grad_{p}': torch.norm(grad[1], p=p)
            }
            all_ret.update(ret)

        # detach and move all the values tensor to cpu
        for k, v in all_ret.items():
            all_ret[k] = rd(v.cpu().detach().numpy().item())

        return all_ret


    def end_post_train(self):
        # end post-training
        self.frozen_entity_embeddings = None
        self.trainable_indices = None
        self.frozen_indices = None
        self.trainable_entity_embeddings = None

    def update_embeddings(self):
        # update embeddings
        self.entity_embeddings[self.trainable_indices] = self.trainable_entity_embeddings

    def all_scores(self, samples: np.array) -> np.array:
        """
            For each of the passed samples, compute scores for all possible tail entities.
            :param samples: a 2-dimensional np array containing the samples to score, one per row
            :return: a 2-dimensional np array that, for each sample, contains a row for each passed sample
                     and a column for each possible tail
        """
        # entity_embeddings, relation_embeddings = self.get_embedding()
        relation_embeddings = self.relation_embeddings

        if self.trainable_indices is not None:
            entity_embeddings = torch.zeros_like(self.entity_embeddings, device='cuda')
            with torch.no_grad():
                try:
                    gpu_lock.acquire()
                    entity_embeddings[self.frozen_indices] = self.frozen_entity_embeddings[self.frozen_indices]
                finally:
                    gpu_lock.release()
            entity_embeddings[self.trainable_indices] = self.trainable_entity_embeddings
        else:
            entity_embeddings  = self.entity_embeddings

        head_embeddings = entity_embeddings[samples[:, 0]]
        relation_embeddings = relation_embeddings[samples[:, 1]]
        
        x = self.calculate_g(head_embeddings, relation_embeddings)
        tail_embeddings = entity_embeddings

        x = torch.mm(x, tail_embeddings.transpose(1, 0))
        #x += self.b.expand_as(x)

        pred = torch.sigmoid(x)

        return pred
    
    def get_tail_set(self, samples, split=' '):
        relations = set(samples[:, 1].tolist())
        # print(len(relations), end=split)
        tail_set = []
        lis = list(set(tail_set))
        lis.sort()
        return lis


    def predict_samples(self, samples: np.array) -> Tuple[Any, Any, Any]:
        """
            This method takes as an input a tensor of 'direct' samples,
            runs head and tail prediction on each of them
            and returns
                - the obtained scores for direct and inverse version of each sample,
                - the obtained head and tail ranks for each sample
                - the list of predicted entities for each sample
            :param samples: a torch.Tensor containing all the DIRECT samples to predict.
                            They will be automatically inverted to perform head prediction

            :return: three dicts mapping each passed direct sample (in Tuple format) respectively to
                        - the scores of that direct sample and of the corresponding inverse sample;
                        - the head and tail rank for that sample;
                        - the head and tail predictions for that sample
        """

        scores, ranks, predictions = [], [], []     # output data structures
        direct_samples = samples

        # assert all samples are direct
        assert (samples[:, 1] < self.dataset.num_direct_relations).all()

        # invert samples to perform head predictions
        inverse_samples = self.dataset.invert_samples(direct_samples)

        #obtain scores, ranks and predictions both for direct and inverse samples
        inverse_scores, head_ranks, head_predictions = self.predict_tails(inverse_samples)
        direct_scores, tail_ranks, tail_predictions = self.predict_tails(direct_samples)

        for i in range(direct_samples.shape[0]):
            # add to the scores list a couple containing the scores of the direct and of the inverse sample
            scores += [(direct_scores[i], inverse_scores[i])]

            # add to the ranks list a couple containing the ranks of the head and of the tail
            ranks.append((int(head_ranks[i]), int(tail_ranks[i])))

            # add to the prediction list a couple containing the lists of predictions
            predictions.append((head_predictions[i], tail_predictions[i]))

        return scores, ranks, predictions

    def get_embedding(self):
        if self.embedding_model:
            entity_embeddings, relation_embeddings =  self.embedding_model(self.dataset.g)
            if self.kelpie_entity_embedding is None:
                return entity_embeddings, relation_embeddings    
            return torch.cat([entity_embeddings, self.kelpie_entity_embedding], 0), relation_embeddings

        return self.entity_embeddings, self.relation_embeddings

    def predict_tails(self, samples: np.array) -> Tuple[Any, Any, Any]:
        """
            Returns filtered scores, ranks and predicted entities for each passed fact.
            :param samples: a torch.LongTensor of triples (head, relation, tail).
                          The triples can also be "inverse triples" with (tail, inverse_relation_id, head)
            :return:
        """

        scores, ranks, pred_out = [], [], []

        batch_size = 128
        for i in range(0, samples.shape[0], batch_size):
            batch = samples[i : min(i + batch_size, len(samples))]

            all_scores = self.all_scores(batch)

            tail_indexes = torch.tensor(batch[:, 2]).cuda()  # tails of all passed samples

            # for each sample to predict
            for sample_number, (head_id, relation_id, tail_id) in enumerate(batch):
                tails_to_filter = self.dataset.to_filter[(head_id, relation_id)]

                # score obtained by the correct tail of the sample
                target_tail_score = all_scores[sample_number, tail_id].item()
                scores.append(target_tail_score)

                # set to 0.0 all the predicted values for all the correct tails for that Head-Rel couple
                all_scores[sample_number, tails_to_filter] = 0.0
                # re-set the predicted value for that tail to the original value
                all_scores[sample_number, tail_id] = target_tail_score

            # this amounts to using ORDINAL policy
            sorted_values, sorted_indexes = torch.sort(all_scores, dim=1, descending=True)
            sorted_indexes = sorted_indexes.cpu().numpy()

            for row in sorted_indexes:
                pred_out.append(row)

            for row in range(batch.shape[0]):
                # rank of the correct target
                rank = np.where(sorted_indexes[row] == tail_indexes[row].item())[0][0]
                ranks.append(rank + 1)

        return scores, ranks, pred_out

    def kelpie_model_class(self):
        return KelpieConvE


class PostConvE(KelpieModel, ConvE):
    def __init__(self, model: ConvE, head: int, init_tensor=None):
        ConvE.__init__(self, model.dataset, model.hyperparameters, False)

        self.head = head
        self.model = model
        if init_tensor is None:
            init_tensor = torch.rand(1, self.dimension)

        # for param in self.parameters():
        #     param.requires_grad = False

        frozen_entity_embeddings = model.entity_embeddings.clone().detach()
        frozen_relation_embeddings = model.relation_embeddings.clone().detach()

        self.trainable_head_embedding = torch.nn.Parameter(init_tensor, requires_grad=True)
        frozen_entity_embeddings[self.head] = self.trainable_head_embedding
        self.entity_embeddings = frozen_entity_embeddings.to('cuda')
        self.relation_embeddings = frozen_relation_embeddings.to('cuda')

        self.convolutional_layer = copy.deepcopy(model.convolutional_layer)
        self.convolutional_layer.requires_grad = False
        self.convolutional_layer.eval()

        self.hidden_layer = copy.deepcopy(model.hidden_layer)
        self.hidden_layer.requires_grad = False
        self.hidden_layer.eval()

        # copy the batchnorms of the original ConvE model and keep them frozen
        self.batch_norm_1 = copy.deepcopy(model.batch_norm_1)  # copy weights and stuff
        self.batch_norm_1.weight.requires_grad = False
        self.batch_norm_1.bias.requires_grad = False
        self.batch_norm_1.eval()

        self.batch_norm_2 = copy.deepcopy(model.batch_norm_2)  # copy weights and stuff
        self.batch_norm_2.weight.requires_grad = False
        self.batch_norm_2.bias.requires_grad = False
        self.batch_norm_2.eval()

        self.batch_norm_3 = copy.deepcopy(model.batch_norm_3)  # copy weights and stuff
        self.batch_norm_3.weight.requires_grad = False
        self.batch_norm_3.bias.requires_grad = False
        self.batch_norm_3.eval()

    def update_embeddings(self):
        with torch.no_grad():
            # self.entity_embeddings[self.kelpie_entity_id] = self.kelpie_entity_embedding
            self.entity_embeddings[self.head] = self.trainable_head_embedding


class KelpieConvE(KelpieModel, ConvE):
    def __init__(
            self,
            dataset: KelpieDataset,
            model: ConvE,
            init_tensor=None):

        ConvE.__init__(self,
                        dataset=dataset,
                        hyperparameters=model.hyperparameters,
                       init_random=False)  # NOTE: this is important! if it is set to True,
                                            # self.entity_embeddings and self.relation_embeddings will be initialized as Parameters
                                            # and it will not be possible to overwrite them with mere Tensors
                                            # such as the one resulting from torch.cat(...) and as frozen_relation_embeddings

        # self.model = model
        self.original_entity_id = dataset.original_entity_id
        self.kelpie_entity_id = dataset.kelpie_entity_id

        # extract the values of the trained embeddings.
        if model.embedding_model:
            frozen_entity_embeddings, frozen_relation_embeddings = model.get_embedding()
        else:
            frozen_entity_embeddings = model.entity_embeddings.clone().detach()
            frozen_relation_embeddings = model.relation_embeddings.clone().detach()

        # the tensor from which to initialize the kelpie_entity_embedding;
        # if it is None it is initialized randomly
        if init_tensor is None:
            init_tensor = torch.rand(len(self.original_entity_id), self.dimension)

        # It is *extremely* important that kelpie_entity_embedding is both a Parameter and an instance variable
        # because the whole approach of the project is to obtain the parameters model params with parameters() method
        # and to pass them to the Optimizer for optimization.
        #
        # If I used .cuda() outside the Parameter, like
        #       self.kelpie_entity_embedding = Parameter(torch.rand(1, 2*self.dimension), requires_grad=True).cuda()
        # IT WOULD NOT WORK because cuda() returns a Tensor, not a Parameter.

        # Therefore kelpie_entity_embedding would not be a Parameter anymore.

        self.kelpie_entity_embedding = Parameter(init_tensor.cuda(), requires_grad=True)
        self.entity_embeddings = torch.cat([frozen_entity_embeddings, self.kelpie_entity_embedding], 0)
        self.relation_embeddings = frozen_relation_embeddings


        self.convolutional_layer = copy.deepcopy(model.convolutional_layer)
        self.convolutional_layer.requires_grad = False
        self.convolutional_layer.eval()

        self.hidden_layer = copy.deepcopy(model.hidden_layer)
        self.hidden_layer.requires_grad = False
        self.hidden_layer.eval()

        # copy the batchnorms of the original ConvE model and keep them frozen
        self.batch_norm_1 = copy.deepcopy(model.batch_norm_1)  # copy weights and stuff
        self.batch_norm_1.weight.requires_grad = False
        self.batch_norm_1.bias.requires_grad = False
        self.batch_norm_1.eval()

        self.batch_norm_2 = copy.deepcopy(model.batch_norm_2)  # copy weights and stuff
        self.batch_norm_2.weight.requires_grad = False
        self.batch_norm_2.bias.requires_grad = False
        self.batch_norm_2.eval()

        self.batch_norm_3 = copy.deepcopy(model.batch_norm_3)  # copy weights and stuff
        self.batch_norm_3.weight.requires_grad = False
        self.batch_norm_3.bias.requires_grad = False
        self.batch_norm_3.eval()

    # Override
    def predict_samples(self,
                        samples: np.array,
                        original_mode: bool = False):
        """
        This method overrides the Model predict_samples method
        by adding the possibility to run predictions in original_mode
        which means ...
        :param samples: the DIRECT samples. Will be inverted to perform head prediction
        :param original_mode:
        :return:
        """

        direct_samples = samples

        # assert all samples are direct
        assert (samples[:, 1] < self.dataset.num_direct_relations).all()

        # if we are in original_mode, make sure that the kelpie entity is not featured in the samples to predict
        # otherwise, make sure that the original entity is not featured in the samples to predict
        forbidden_entity_id = self.kelpie_entity_id if original_mode else self.original_entity_id
        assert np.isin(forbidden_entity_id, direct_samples[:][0, 2]) == False

        # use the ConvE implementation method to obtain scores, ranks and prediction results.
        # these WILL feature the forbidden entity, so we now need to filter them
        scores, ranks, predictions = ConvE.predict_samples(self, direct_samples)

        # remove any reference to the forbidden entity id
        # (that may have been included in the predicted entities)
        for i in range(len(direct_samples)):
            head_predictions, tail_predictions = predictions[i]
            head_rank, tail_rank = ranks[i]

            # remove the forbidden entity id from the head predictions (note: it could be absent due to filtering)
            # and if it was before the head target decrease the head rank by 1
            forbidden_indices = np.where(head_predictions == forbidden_entity_id)[0]
            if len(forbidden_indices) > 0:
                index = forbidden_indices[0]
                head_predictions = np.concatenate([head_predictions[:index], head_predictions[index + 1:]], axis=0)
                if index < head_rank:
                    head_rank -= 1

            # remove the kelpie entity id from the tail predictions  (note: it could be absent due to filtering)
            # and if it was before the tail target decrease the head rank by 1
            forbidden_indices = np.where(tail_predictions == forbidden_entity_id)[0]
            if len(forbidden_indices) > 0:
                index = forbidden_indices[0]
                tail_predictions = np.concatenate([tail_predictions[:index], tail_predictions[index + 1:]], axis=0)
                if index < tail_rank:
                    tail_rank -= 1

            predictions[i] = (head_predictions, tail_predictions)
            ranks[i] = (head_rank, tail_rank)

        return scores, ranks, predictions


    # Override
    def predict_sample(self,
                       sample: np.array,
                       original_mode: bool = False):
        """
        Override the
        :param sample: the DIRECT sample. Will be inverted to perform head prediction
        :param original_mode:
        :return:
        """

        assert sample[1] < self.dataset.num_direct_relations

        scores, ranks, predictions = self.predict_samples(np.array([sample]), original_mode)
        return scores[0], ranks[0], predictions[0]