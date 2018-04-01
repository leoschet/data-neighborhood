from enum import Enum
import codecs
import arff
import os

class EDataType(Enum):
    """
    Enumerable class with all data types
    """
    NUMERICAL = 'numerical'
    CATEGORICAL = 'categorical'
    MIXED = 'mixed'

class DataCollection(object):
    def __init__(self, corpus_dir):
        self._corpus_dir = corpus_dir
        self.documents = self._get_documents()

    def get_data(self, relation):
        relation_data = self.documents[relation]['data']
        data = [(data[:(len(data) - 1)], data[(len(data) - 1):][0]) for data in relation_data]
        return data

    def _get_documents(self):
        files_name = os.listdir(self._corpus_dir)
        file_relation = [self._get_document(self._corpus_dir + file_name) for file_name in files_name
                 if file_name.endswith(".arff")]

        files = {}
        for file_, relation in file_relation:
            assert relation not in files
            files[relation] = file_

        return files

    def _get_document(self, path):
        file_ = codecs.open(path, 'rb', 'utf-8')
        data = arff.load(file_)
        return (data, data['relation'])
