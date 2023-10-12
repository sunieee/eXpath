from typing import Tuple, Any
from dataset import Dataset
from relevance_engines.criage_engine import CriageEngine
from link_prediction.models.model import Model
from prefilters.criage_prefilter import CriagePreFilter
from explanation_builders.criage_necessary_builder import CriageNecessaryExplanationBuilder

class Criage:
    """
    The Criage object is the overall manager of the Criage explanation process.
    It implements the whole explanation pipeline, requesting the suitable operations to the ExplanationEngines
    and to the entity_similarity modules.
    """

    def __init__(self,
                 model: Model,
                 dataset: Dataset,
                 hyperparameters: dict):
        """
        Criage object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        :param hyperparameters: the hyperparameters of the model and of its optimization process
        """
        self.model = model
        self.dataset = dataset
        self.hyperparameters = hyperparameters

        self.prefilter = CriagePreFilter(model=model,
                                         dataset=dataset)
        self.engine = CriageEngine(model=model,
                                   dataset=dataset,
                                   hyperparameters=hyperparameters)

    def explain_necessary(self,
                          sample_to_explain: Tuple[Any, Any, Any],
                          perspective: str,
                          num_promising_samples=-1):
        """
        This method extracts necessary explanations for a specific sample,
        from the perspective of either its head or its tail.

        :param sample_to_explain: the sample to explain
        :param perspective: a string conveying the perspective of the requested explanations.
                            It can be either "head" or "tail":
                                - if "head", Kelpie answers the question
                                    "given the sample head and relation, why is the sample tail predicted as tail?"
                                - if "tail", Kelpie answers the question
                                    "given the sample relation and tail, why is the sample head predicted as head?"
        :param num_promising_samples: the number of samples relevant to the sample to explain
                                     that must be identified and removed from the entity under analysis
                                     to verify whether they worsen the target prediction or not

        :return: a list containing for each relevant n-ple extracted, a couple containing
                                - that relevant n-ple
                                - its value of relevance

        """

        top_promising_samples = self.prefilter.top_promising_explanations(sample_to_explain=sample_to_explain,
                                                                         perspective=perspective,
                                                                         top_k=num_promising_samples)

        explanation_builder = CriageNecessaryExplanationBuilder(model=self.model,
                                                                dataset=self.dataset,
                                                                hyperparameters=self.hyperparameters,
                                                                sample_to_explain=sample_to_explain,
                                                                perspective=perspective)

        rules_with_relevance = explanation_builder.build_explanations(samples_to_remove=top_promising_samples)
        return rules_with_relevance

