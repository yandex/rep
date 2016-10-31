"""
This class is used to build predictions of MatrixNet classifier
it uses .mx format of MatrixNet formula.
"""
from __future__ import print_function, division, absolute_import
import struct
import numpy

__author__ = 'Alex Rogozhnikov, Egor Khairullin'


class MatrixNetApplier(object):
    def __init__(self, formula_stream):
        self.features = []  # list of strings
        self.bins = []

        features_quantity = struct.unpack('i', formula_stream.read(4))[0]
        for index in range(0, features_quantity):
            factor_length = struct.unpack('i', formula_stream.read(4))[0]
            self.features.append(formula_stream.read(factor_length))

        _ = formula_stream.read(4)  # skip formula length
        used_features_quantity = struct.unpack('I', formula_stream.read(4))[0]
        bins_quantities = self.read_array(formula_stream, 'I', used_features_quantity)

        self.bins_total = struct.unpack('I', formula_stream.read(4))[0]
        for index in range(used_features_quantity):
            self.bins.append(self.read_array(formula_stream, 'f', bins_quantities[index]))

        _ = formula_stream.read(4)  # skip classes_count == 0

        nf_counts_len = struct.unpack('I', formula_stream.read(4))[0]
        self.nf_counts = self.read_array(formula_stream, 'I', nf_counts_len)

        ids_len = struct.unpack('I', formula_stream.read(4))[0]
        self.feature_ids = self.read_array(formula_stream, 'I', ids_len)

        tree_table_len = struct.unpack('I', formula_stream.read(4))[0]
        self.tree_table = self.read_array(formula_stream, 'i', tree_table_len)

        self.bias = struct.unpack('d', formula_stream.read(8))[0]
        self.delta_mult = struct.unpack('d', formula_stream.read(8))[0]

    @staticmethod
    def read_array(stream, element_formatter, length):
        elements_length = {'d': 8, 'i': 4, 'I': 4, 'f': 4}
        array_format = '{}{}'.format(length, element_formatter)
        array_size_in_bytes = elements_length[element_formatter] * length
        # checking that sizes are matching
        assert struct.calcsize(array_format) == array_size_in_bytes

        result = struct.unpack(array_format, stream.read(array_size_in_bytes))
        return numpy.array(result)

    def get_stats(self):
        """
        Function returns different information about this formula as a dict
        """
        stats_dict = {}
        stats_dict['bias'] = self.bias
        stats_dict['bins(lenghts)'] = [len(x) for x in self.bins]
        stats_dict['total_bins'] = self.bins_total
        stats_dict['features'] = self.features
        stats_dict['delta_mult'] = self.delta_mult
        stats_dict['len(feature_ids)'] = len(self.feature_ids)
        stats_dict['nf_counts'] = self.nf_counts
        stats_dict['len(tree_table)'] = len(self.tree_table)

        return stats_dict

    def _prepare_features_and_cuts(self):
        """
        Returns tuple with information about binary features:
            numpy.array of shape [n_binary_features] with indices of (initial) features used
            numpy.array of shape [n_binary_features] with cuts used (float32)
        """
        n_binary_features = sum(len(x) for x in self.bins)
        features_ids = numpy.zeros(n_binary_features, dtype='uint64')
        cuts = numpy.zeros(n_binary_features, dtype='float32')
        binary_feature_index = 0
        for feature_index in range(len(self.bins)):
            for cut in self.bins[feature_index]:
                features_ids[binary_feature_index] = feature_index
                cuts[binary_feature_index] = cut
                binary_feature_index += 1
        return features_ids, cuts

    def _prepare_2d_features_and_cuts(self, binary_feature_ids):
        """
        Provided indices of binary features used in trees, returns in the same format
        :param binary_feature_ids:
            numpy.array of shape [n_trees, tree_depth]
        :return: tuple,
            numpy.array of shape [n_trees, tree_depth] with indices of (initial features) used
            numpy.array of shape [n_trees, tree_depth] with cuts used
        """
        feature_ids, cuts = self._prepare_features_and_cuts()
        return feature_ids[binary_feature_ids], cuts[binary_feature_ids]

    def _iterate_over_trees_with_fixed_depth(self, tree_depth, n_trees,
                                             binary_features_index, leaves_table_index):

        leaves_in_tree = 1 << tree_depth
        binary_feature_ids = self.feature_ids[binary_features_index:binary_features_index + n_trees * tree_depth]
        binary_feature_ids = binary_feature_ids.reshape([-1, tree_depth])

        feature_ids, cuts = self._prepare_2d_features_and_cuts(binary_feature_ids)
        for tree in range(n_trees):
            leaf_values = self.tree_table[leaves_table_index:leaves_table_index + leaves_in_tree] / self.delta_mult
            leaves_table_index += leaves_in_tree
            yield feature_ids[tree, :], cuts[tree, :], leaf_values

    def iterate_trees(self):
        """
        :return: yields depth, n_trees, trees_iterator,
            trees_iterator yields feature_ids, feature_cuts, leaf_values
        """
        binary_features_index = 0
        leaves_table_index = 0

        for tree_depth, n_trees in enumerate(self.nf_counts, 1):
            if n_trees == 0:
                continue

            yield tree_depth, n_trees, self._iterate_over_trees_with_fixed_depth(
                    tree_depth=tree_depth, n_trees=n_trees,
                    binary_features_index=binary_features_index,
                    leaves_table_index=leaves_table_index)
            binary_features_index += tree_depth * n_trees
            leaves_table_index += (1 << tree_depth) * n_trees

    def apply_separately(self, events):
        """
        :param events: numpy.array (or DataFrame) of shape [n_samples, n_features]
        :return: each time yields numpy.array predictions of shape [n_samples]
            which is output of a particular tree
        """
        # result of first iteration
        yield numpy.zeros(len(events), dtype=float) + self.bias

        # extending the data so the number of events is divisible by 8
        n_events = len(events)
        n_extended64 = (n_events + 7) // 8
        n_extended = n_extended64 * 8

        # using Fortran order (surprisingly doesn't seem to influence speed much)
        features = numpy.zeros([n_extended, events.shape[1]], dtype='float32', order='F')
        features[:n_events, :] = events

        for tree_depth, nf_count, tree_iterator in self.iterate_trees():
            for tree_features, tree_cuts, leaf_values in tree_iterator:
                leaf_indices = numpy.zeros(n_extended64, dtype='uint64')
                for tree_level, (feature, cut) in enumerate(zip(tree_features, tree_cuts)):
                    leaf_indices |= (features[:, feature] > cut).view('uint64') << tree_level
                yield leaf_values[leaf_indices.view('uint8')[:n_events]]

    def apply(self, events):
        """
        :param events: numpy.array (or DataFrame) of shape [n_samples, n_features]
        :return: prediction of shape [n_samples]
        """
        result = numpy.zeros(len(events), dtype=float)
        for stage_predictions in self.apply_separately(events):
            result += stage_predictions
        return result

    def compute_leaf_indices_separately(self, events):
        """ For each tree yields leaf_indices of events """
        # extending the data so the number of events is divisible by 8
        n_events = len(events)
        n_extended64 = (n_events + 7) // 8
        n_extended = n_extended64 * 8

        # using Fortran order (surprisingly doesn't seem to influence speed much)
        features = numpy.zeros([n_extended, events.shape[1]], dtype='float32', order='F')
        features[:n_events, :] = events

        for tree_depth, n_trees, tree_iterator in self.iterate_trees():
            for tree_features, tree_cuts, leaf_values in tree_iterator:
                leaf_indices = numpy.zeros(n_extended, dtype='uint8')
                leaf_indices_view = leaf_indices.view('uint64')
                for tree_level, (feature, cut) in enumerate(zip(tree_features, tree_cuts)):
                    leaf_indices_view |= (features[:, feature] > cut).view('uint64') << tree_level
                yield leaf_indices[:n_events]

    def compute_leaf_indices(self, events):
        """
        :param events: pandas.DataFrame of shape [n_events, n_features]
        :return: numpy.array of shape [n_events, n_trees]
        """
        result = numpy.zeros([len(events), sum(self.nf_counts)], dtype='uint8', order='F')
        for tree, tree_leaves in enumerate(self.compute_leaf_indices_separately(events)):
            result[:, tree] = tree_leaves
        return result
