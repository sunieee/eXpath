import numpy
from link_prediction.evaluation.evaluation import Evaluator
from link_prediction.models.model import Model
import torch
import numpy as np


class Optimizer:
    """
        The Optimizer class provides the interface that any LP Optimizer should implement.
    """

    def __init__(self,
                 model: Model,
                 hyperparameters: dict,
                 verbose: bool = True):

        self.model = model  #type:Model
        self.dataset = self.model.dataset
        self.verbose = verbose

        # create the evaluator to use between epochs
        self.evaluator = Evaluator(self.model)

    def train(self,
          train_samples: np.array,
          save_path: str = None,
          evaluate_every: int = -1,
          valid_samples: np.array = None,
          patience: int = -1,
          type='tail'):

        training_samples = np.vstack((train_samples, 
                                      self.model.dataset.invert_samples(train_samples)))
        batch_size = min(self.batch_size, len(training_samples))
        self.model.cuda()

        best_metric = None
        best_epoch = None
        deteriorating_epochs = 0
        has_save = False

        for e in range(1, self.epochs + 1):
            self.epoch(batch_size, training_samples, epoch=e)

            if evaluate_every > 0 and valid_samples is not None and e % evaluate_every == 0:
                self.model.eval()
                with torch.no_grad():
                    mrr, h1, h5, h10, mr = self.evaluator.evaluate(samples=valid_samples, 
                                                                   write_output=False,
                                                                   type=type)
                self.model.train()

                if patience == -1:
                    continue

                if best_metric is not None and mrr < best_metric:
                    deteriorating_epochs += 1
                    if deteriorating_epochs >= patience:
                        print("Early stopping triggered. Best metric occurred at epoch", best_epoch)
                        break
                else:
                    deteriorating_epochs = 0

                if best_metric is None or mrr > best_metric:
                    best_metric = mrr
                    best_epoch = e
                    if save_path is not None:
                        print("Saving model at epoch", best_epoch)
                        torch.save(self.model.state_dict(), save_path)
                        has_save = True

        if not has_save and save_path is not None:
            torch.save(self.model.state_dict(), save_path)


        
    # def epoch(self,
    #          batch_size: int,
    #          training_samples: numpy.array):
    #    pass

    # def step_on_batch(self, loss, batch):
    #    pass


