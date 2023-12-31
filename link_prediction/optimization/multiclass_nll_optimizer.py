import torch
import tqdm
import numpy as np
from torch import optim, nn

from link_prediction.regularization.regularizers import N3, N2
from link_prediction.models.model import Model, OPTIMIZER_NAME, BATCH_SIZE, EPOCHS, LEARNING_RATE, DECAY_1, REGULARIZER_NAME, \
    REGULARIZER_WEIGHT, DECAY_2, KelpieModel
from link_prediction.optimization.optimizer import Optimizer

class MultiClassNLLOptimizer(Optimizer):
    """
        This optimizer relies on Multiclass Negative Log Likelihood loss.
        It is heavily inspired by paper ""
        Instead of considering each training sample as the "unit" for training,
        it groups training samples into couples (h, r) -> [all t for which <h, r, t> in training set].
        Each couple (h, r) with the corresponding tails is treated as if it was one sample.

        When passing them to the loss...

        In our implementation, it is used by the following models:
            - ComplEx
            - DistMult

    """

    def __init__(self,
                 model: Model,
                 hyperparameters: dict,
                 verbose: bool = True):

        Optimizer.__init__(self, model=model, hyperparameters=hyperparameters, verbose=verbose)

        self.args = model.dataset.args
        self.optimizer_name = hyperparameters[OPTIMIZER_NAME]
        self.batch_size = hyperparameters[BATCH_SIZE]
        self.epochs = hyperparameters[EPOCHS]
        self.learning_rate = hyperparameters[LEARNING_RATE]
        self.decay1, self.decay2 = hyperparameters[DECAY_1], hyperparameters[DECAY_2]
        self.regularizer_name = hyperparameters[REGULARIZER_NAME]
        self.regularizer_weight = hyperparameters[REGULARIZER_WEIGHT]

        # build all the supported optimizers using the passed params (learning rate and decays if Adam)
        supported_optimizers = {
            'Adagrad': optim.Adagrad(params=self.model.parameters(), lr=self.learning_rate),
            'Adam': optim.Adam(params=self.model.parameters(), lr=self.learning_rate, betas=(self.decay1, self.decay2)),
            'SGD': optim.SGD(params=self.model.parameters(), lr=self.learning_rate)
        }

        # build all the supported regularizers using the passed regularizer_weight
        supported_regularizers = {
            'N3': N3(weight=self.regularizer_weight),
            'N2': N2(weight=self.regularizer_weight)
        }

        # choose the Torch Optimizer object to use, based on the passed name
        self.optimizer = supported_optimizers[self.optimizer_name]

        # choose the regularizer
        self.regularizer = supported_regularizers[self.regularizer_name]

    def epoch(self,
              batch_size: int,
              training_samples: np.array, epoch: int=0):
        training_samples = torch.from_numpy(training_samples).cuda()

        # at the beginning of the epoch, shuffle all samples randomly
        actual_samples = training_samples[torch.randperm(training_samples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')

        with tqdm.tqdm(total=training_samples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'epoch {epoch} loss')

            batch_start = 0
            while batch_start < training_samples.shape[0]:
                batch_end = min(batch_start + batch_size, training_samples.shape[0])
                batch = actual_samples[batch_start : batch_end].cuda()
                l = self.step_on_batch(loss, batch)

                batch_start += self.batch_size
                bar.update(batch.shape[0])
                bar.set_postfix(loss=f'{l.item():.4f}')


    def step_on_batch(self, loss, batch):
        predictions, factors = self.model.forward(batch)
        truth = batch[:, 2]

        # compute loss
        l_fit = loss(predictions, truth)
        l_reg = self.regularizer.forward(factors)
        l = l_fit + l_reg

        # compute loss gradients, and run optimization step
        self.optimizer.zero_grad()
        l.backward()
        self.optimizer.step()

        # return loss
        return l


class KelpieMultiClassNLLOptimizer(MultiClassNLLOptimizer):
    def __init__(self,
                 model: KelpieModel,
                 hyperparameters: dict,
                 verbose: bool = True):
        MultiClassNLLOptimizer.__init__(self,
                                        model=model,
                                        hyperparameters=hyperparameters,
                                        verbose=verbose)

    def epoch(self,
              batch_size: int,
              training_samples: np.array, epoch: int=0):
        training_samples = torch.from_numpy(training_samples).cuda()
        # at the beginning of the epoch, shuffle all samples randomly
        actual_samples = training_samples[torch.randperm(training_samples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean')

        with tqdm.tqdm(total=training_samples.shape[0], unit='ex', disable=not self.verbose) as bar:
            bar.set_description(f'epoch {epoch} loss')

            batch_start = 0
            while batch_start < training_samples.shape[0]:
                batch = actual_samples[batch_start: batch_start + batch_size].cuda()
                l = self.step_on_batch(loss, batch)

                # THIS IS THE ONE DIFFERENCE FROM THE ORIGINAL OPTIMIZER.
                # THIS IS EXTREMELY IMPORTANT BECAUSE THIS WILL PROPAGATE THE UPDATES IN THE KELPIE ENTITY EMBEDDING
                # TO THE MATRIX CONTAINING ALL THE EMBEDDINGS
                self.model.update_embeddings()

                batch_start += self.batch_size
                bar.update(batch.shape[0])
                bar.set_postfix(loss='{l.item():.0f}')