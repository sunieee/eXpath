from typing import Tuple, Any
from dataset import Dataset
from explanation_builders.prefilter import TopologyPreFilter
from relevance_engines.data_poisoning_engine import DataPoisoningEngine
from link_prediction.models.model import Model, LEARNING_RATE
from explanation_builders.dp_builder import DataPoisoningNecessaryExplanationBuilder


class DataPoisoning:
    """
    The DataPoisoning object is the overall manager of the Data_poisoning explanation process.
    It implements the whole explanation pipeline, requesting the suitable operations to the ExplanationEngines
    and to the entity_similarity modules.
    """

    def __init__(self,
                 model: Model,
                 dataset: Dataset,
                 hyperparameters: dict):
        """
        DataPoisoning object constructor.

        :param model: the model to explain
        :param dataset: the dataset used to train the model
        :param hyperparameters: the hyperparameters of the model and of its optimization process
        :param prefilter_type: the type of prefilter to employ
        """
        self.model = model
        self.dataset = dataset
        self.hyperparameters = hyperparameters

        self.prefilter = TopologyPreFilter(model=model, dataset=dataset)
        self.engine = DataPoisoningEngine(model=model,
                                          dataset=dataset,
                                          hyperparameters=hyperparameters,
                                          epsilon=hyperparameters[LEARNING_RATE])

    def explain(self,
                          sample_to_explain: Tuple[Any, Any, Any],
                          perspective: str,
                          num_promising_samples=50):
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

        most_promising_samples = self.prefilter.top_promising_explanations(sample_to_explain=sample_to_explain,
                                                                          perspective=perspective,
                                                                          top_k=num_promising_samples)

        rule_extractor = DataPoisoningNecessaryExplanationBuilder(model=self.model,
                                                                  dataset=self.dataset,
                                                                  hyperparameters=self.hyperparameters,
                                                                  sample_to_explain=sample_to_explain,
                                                                  perspective=perspective, )

        rules_with_relevance = rule_extractor.build_explanations(samples_to_remove=most_promising_samples)
        return rules_with_relevance
