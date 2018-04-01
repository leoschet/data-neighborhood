from ..retriever.data_collection import EDataType

class KNeighborsClassifier(object):
    def __init__(self, k, train_data, data_type, weighted_distance=True):
        self.k = k
        self.weighted_distance = weighted_distance
        
        self._train_data = train_data
        self._data_type = EDataType(data_type)

    def classify(self, in_features):
        k_nearest = self._get_nearest(in_features) # They come as a tuple: (distance, label)

        possible_labels = {}
        best_label = None
        # print ('\ntesting:', k_nearest)
        for distance, label in k_nearest:
            if label not in possible_labels:
                possible_labels[label] = 0

            if (self.weighted_distance):
                possible_labels[label] += distance
            else:
                possible_labels[label] += 1

            # In case possible_labels[label] == possible_labels[best_label], then
            # choose randomly the label between possibilities
            if best_label is None or possible_labels[label] > possible_labels[best_label]:
                best_label = label

        return possible_labels, best_label

    def _get_nearest(self, in_features):
        """
        Returns an array of tuples (distance, label) with the k-nearest neighbors from in_features.
        """
        labeled_distances = []
        for features, label in self._train_data:
            distance = self._calculate_distance(in_features, features)
            labeled_distances.append((distance, label))
        
        labeled_distances.sort(key=lambda tup: tup[0], reverse=self.weighted_distance)
        # print ('testing:', labeled_distances)        
        return labeled_distances[:self.k]

    def _calculate_distance(self, in_features, features):
        distance = -1

        if self._data_type == EDataType.NUMERICAL:
            distance = self._euclidean_distance(in_features, features)
        elif self._data_type == EDataType.CATEGORICAL:
            distance = self._vdm_distance(in_features, features)
        elif self._data_type == EDataType.MIXED:
            distance = self._hvdm_distance(in_features, features)

        if distance == 0:
            distance = 1

        if (self.weighted_distance):
            distance = 1/(distance ** 2)

        # TODO: Check better way to treat error case
        return distance

    def _euclidean_distance(self, in_features, features):
        assert len(in_features) == len(features)

        distance = 0
        for index, feature in enumerate(features):
            distance += (feature - in_features[index]) ** 2

        return distance ** (1/2)

    def _vdm_distance(self, in_features, features):
        return -1

    def _hvdm_distance(self, in_features, features):
        return -1

