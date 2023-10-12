import copy
from collections import defaultdict
import numpy
from dataset import Dataset

class XruleDataset(Dataset):
    """
        Since Datasets handle the correspondence between textual entities and ids,
        the KelpieDataset has the responsibility to decide the id of the kelpie entity (aka mimic in our paper)
        and to store the train, valid and test samples specific for the original entity and for the kelpie entity

        A KelpieDataset is never *loaded* from file: it is always generated from a pre-existing, already loaded Dataset.

        Nomenclature used in the KelpieDataset:
            "original entity": the entity to explain the prediction of in the original Dataset;
            "kelpie entity": a (non-)homologous mimic, i.e., a "fake" entity
                               post-trained with slightly different training samples from the original entity.
                               (e.g. some training samples may have been removed, or added).
    """

    def __init__(self,
                 dataset: Dataset,
                 original_entity_id): 
        # 所有与entity_ids相关的训练样本（包括反向关系）

        super(XruleDataset, self).__init__(name=dataset.name,
                                            separator=dataset.separator,
                                            load=False,
                                            args=dataset.args)
        if dataset.num_entities == -1:
            raise Exception("The Dataset passed to initialize a KelpieDataset must be already loaded")

        # the KelpieDataset is now basically empty (because load=False was used in the super constructor)
        # so we must manually copy (and sometimes update) all the important attributes from the original loaded Dataset
        if hasattr(original_entity_id, '__iter__'):
            self.original_entity_id = original_entity_id
        else:
            self.original_entity_id = [original_entity_id]
        
        print('\tkelpie init: entity_id:', self.original_entity_id)
        self.l = len(self.original_entity_id)
        self.num_entities = dataset.num_entities + self.l                 # adding B & C to the count
        self.num_relations = dataset.num_relations
        self.num_direct_relations = dataset.num_direct_relations
        self.dataset = dataset

        # copy relevant data structures
        self.to_filter = copy.deepcopy(dataset.to_filter)
        self.train_to_filter = copy.deepcopy(dataset.train_to_filter)
        self.entity_name_2_id = copy.deepcopy(dataset.entity_name_2_id)
        self.entity_id_2_name = copy.deepcopy(dataset.entity_id_2_name)
        self.relation_name_2_id = copy.deepcopy(dataset.relation_name_2_id)
        self.relation_id_2_name = copy.deepcopy(dataset.relation_id_2_name)
        self.rid2target = copy.deepcopy(dataset.rid2target)
        if dataset.g:
            # https://stackoverflow.com/questions/60310123/why-is-graph-copy-slower-than-copy-deepcopygraph-in-networkx
            self.g = copy.deepcopy(dataset.g)

        # add the kelpie entity
        self.original_entity_id = original_entity_id
        self.kelpie_entity_id = []     # (non-)homologous mimic * l
        self.entity2kelpie = {}

        self.kelpie_train_samples = []
        self.kelpie_valid_samples = []
        self.kelpie_test_samples = []

        for ix, original_entity_id in enumerate(self.original_entity_id):
            entity_name = self.entity_id_2_name[original_entity_id]
            kelpie_name = "kelpie_" + entity_name
            kelpie_id = dataset.num_entities + ix
            self.entity_name_2_id[kelpie_name] = kelpie_id
            self.entity_id_2_name[kelpie_id] = kelpie_name
            self.kelpie_entity_id.append(kelpie_id)
            self.entity2kelpie[original_entity_id] = kelpie_id


        self.original_train_samples = self._extract_samples_with_entities(dataset.train_samples, self.original_entity_id)
        self.original_valid_samples = self._extract_samples_with_entities(dataset.valid_samples, self.original_entity_id)
        self.original_test_samples = self._extract_samples_with_entities(dataset.test_samples, self.original_entity_id)

        self.kelpie_train_samples = Dataset.replace_entities_in_samples(self.original_train_samples, self.original_entity_id, self.kelpie_entity_id)
        self.kelpie_valid_samples = Dataset.replace_entities_in_samples(self.original_valid_samples, self.original_entity_id, self.kelpie_entity_id)
        self.kelpie_test_samples = Dataset.replace_entities_in_samples(self.original_test_samples, self.original_entity_id, self.kelpie_entity_id)

        print('\tent2kelpie:', self.entity2kelpie)
        print(f'\tkelpie dataset created: train {len(self.kelpie_train_samples)}, valid {len(self.kelpie_valid_samples)}, test {len(self.kelpie_test_samples)}')
        # print(self.kelpie_train_samples)
        # print('kelpie train samples:', self.kelpie_train_samples[:10])

        # update to_filter and train_to_filter data structures
        samples_to_stack = [self.kelpie_train_samples]
        if len(self.kelpie_valid_samples) > 0:
            samples_to_stack.append(self.kelpie_valid_samples)
        if len(self.kelpie_test_samples) > 0:
            samples_to_stack.append(self.kelpie_test_samples)
        all_kelpie_samples = numpy.vstack(samples_to_stack)
        for i in range(all_kelpie_samples.shape[0]):
            self.append_sample(self.to_filter, all_kelpie_samples[i])
            # if the sample was a training sample, also do the same for the train_to_filter data structure;
            # Also fill the entity_2_degree and relation_2_degree dicts.
            if i < len(self.kelpie_train_samples):
                self.append_sample(self.train_to_filter, all_kelpie_samples[i])

        # create a map that associates each kelpie train_sample to its index in self.kelpie_train_samples
        # this will be necessary to allow efficient removals and undoing removals
        self.kelpie_train_sample_2_index = {}
        for i in range(len(self.kelpie_train_samples)):
            cur_head, cur_rel, cur_tail = self.kelpie_train_samples[i]
            self.kelpie_train_sample_2_index[(cur_head, cur_rel, cur_tail)] = i

        # initialize data structures needed in the case of additions and/or removals;
        # these structures are required to undo additions and/or removals
        self.kelpie_train_samples_copy = copy.deepcopy(self.kelpie_train_samples)

        self.last_added_samples = []
        self.last_added_samples_number = 0
        self.last_filter_additions = defaultdict(lambda:[])
        self.last_added_kelpie_samples = []

        self.last_removed_samples = []
        self.last_removed_samples_number = 0
        self.last_filter_removals = defaultdict(lambda:[])
        self.last_removed_kelpie_samples = []

    def __len__(self):
        return len(self.kelpie_train_samples)

    def assert_have_entities(self, sample, entity_id=None):
        if entity_id is None:
            entity_id = self.original_entity_id
        assert sample[0] in entity_id or sample[2] in entity_id
        # if not self.head_id in original_sample:
        #     raise Exception("Could not find the original entity " + str(self.head_id) + " in the passed sample " + str(original_sample))s

    # override
    def remove_training_samples(self, samples_to_remove: numpy.array):
        """
            Remove some training samples from the kelpie training samples of this KelpieDataset.
            The samples to remove must still feature the original entity id; this method will convert them before removal.
            The KelpieDataset will keep track of the last performed removal so it can be undone if necessary.

            :param samples_to_remove: the samples to add, still featuring the id of the original entity,
                                   in the form of a numpy array
        """

        for sample in samples_to_remove:
            self.assert_have_entities(sample)

        self.last_removed_samples = samples_to_remove
        self.last_removed_samples_number = len(samples_to_remove)

        # reset data structures needed to undo removals. We only want to keep track of the *last* removal.
        self.last_filter_removals = defaultdict(lambda:[])
        self.last_removed_kelpie_samples = []

        kelpie_train_samples_to_remove = Dataset.replace_entities_in_samples(samples_to_remove, self.original_entity_id, self.kelpie_entity_id, as_numpy=False)

        print('removing kelpie smaples:', kelpie_train_samples_to_remove)
        # update to_filter and train_to_filter
        for sample in kelpie_train_samples_to_remove:
            self.remove_sample(self.to_filter, sample)
            self.remove_sample(self.train_to_filter, sample)
            self.last_removed_kelpie_samples.append(sample)
            self.append_sample(self.last_filter_removals, sample)

        # get the indices of the samples to remove in the kelpie_train_samples structure
        # and use them to perform the actual removal
        kelpie_train_indices_to_remove = [self.kelpie_train_sample_2_index[x] for x in kelpie_train_samples_to_remove]
        self.kelpie_train_samples = numpy.delete(self.kelpie_train_samples, kelpie_train_indices_to_remove, axis=0)

    
    def remove_sample(self, data, kelpie_sample):
        # 从data中去除kelpie_sample（及其逆关系）
        h, r, t = kelpie_sample
        assert r < self.num_direct_relations
        try:
            data[(h, r)].remove(t)
            data[(t, r + self.num_direct_relations)].remove(h)
        except Exception as e:
            print('!remove sample error:', kelpie_sample)
            print(e)

        # if r < self.num_direct_relations:
        #     data[(t, r + self.num_direct_relations)].remove(h)
        # else:
        #     data[(t, r - self.num_direct_relations)].remove(h)

    def append_sample(self, data, kelpie_sample):
        # 从data中去除kelpie_sample（及其逆关系）
        h, r, t = kelpie_sample
        assert r < self.num_direct_relations
        data[(h, r)].append(t)
        data[(t, r + self.num_direct_relations)].append(h)


    def undo_last_training_samples_removal(self):
        """
            This method undoes the last removal performed on this KelpieDataset
            calling its add_training_samples method.

            The purpose of undoing the removals performed on a pre-existing KelpieDataset,
            instead of creating a new KelpieDataset from scratch, is to improve efficiency.
        """
        if self.last_removed_samples_number <= 0:
            log("No removal to undo. Skip!")
            self.last_removed_samples = []
            self.last_removed_samples_number = 0
            self.last_filter_removals = defaultdict(lambda:[])
            self.last_removed_kelpie_samples = []
            return

        # revert the self.kelpie_train_samples to the self.kelpie_train_samples_copy
        self.kelpie_train_samples = copy.deepcopy(self.kelpie_train_samples_copy)

        # undo additions to to_filter and train_to_filter
        for key in self.last_filter_removals:
            for x in self.last_filter_removals[key]:
                self.to_filter[key].append(x)
                self.train_to_filter[key].append(x)

        # reset the data structures used to undo additions
        self.last_removed_samples = []
        self.last_removed_samples_number = 0
        self.last_filter_removals = defaultdict(lambda:[])
        self.last_removed_kelpie_samples = []

    def as_kelpie_sample(self, original_sample):
        self.assert_have_entities(original_sample)
        for entity_id in self.original_entity_id:
            original_sample = Dataset.replace_entity_in_sample(sample=original_sample,
                                                old_entity=entity_id,
                                                new_entity=self.entity2kelpie[entity_id])
        return original_sample

    ### private utility methods
    @staticmethod
    def _extract_samples_with_entity(samples, entity_id):
        result = samples[numpy.where(numpy.logical_or(samples[:, 0] == entity_id, samples[:, 2] == entity_id))]

        # Check if the result is 1D, and if so, convert to 2D
        if result.ndim == 1:
            result = result[numpy.newaxis, :]

        assert result.ndim == 2
        return result
    
    @staticmethod
    def _extract_samples_with_entities(samples, entity_ids):
        result = samples[numpy.where(numpy.logical_or(numpy.isin(samples[:, 0], entity_ids),
                                                      numpy.isin(samples[:, 2], entity_ids)))]

        # Check if the result is 1D, and if so, convert to 2D
        if result.ndim == 1:
            result = result[numpy.newaxis, :]

        assert result.ndim == 2
        return result