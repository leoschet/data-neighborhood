from app.retriever.data_collection import DataCollection, EDataType
from app.classifier.knn import KNeighborsClassifier

from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.exceptions import UndefinedMetricWarning
import warnings

import numpy as np

import codecs

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

print('Collecting data...')
data_collection = DataCollection('../res/mixed/')

kf = KFold(n_splits=5, shuffle=True) # 80% for training, 20% for testing
ks = [1, 2, 3, 4, 5, 7, 9, 11, 13, 15]
# ks = [5]

verbose = 3 # Can be 0, 1, 2 or 3

for relation in data_collection.documents:
    file_ = codecs.open('../res/results/' + relation + '.txt', 'w+', 'utf-8')

    print('\nTesting for ' + relation + ' data set')
    relation_data, relation_labels = data_collection.get_data_label(relation)
    data_len = len(relation_data)
    print('\tTotal data collected: ' + str(data_len))

    features_types = data_collection.get_features_types(relation)

    for k in ks:
        knn = KNeighborsClassifier(k, weighted_distance=True)
        
        metrics = []
        for train_indexes, test_indexes in kf.split(relation_data):
            train_data, test_data = relation_data[train_indexes], relation_data[test_indexes]
            train_labels, test_labels = relation_labels[train_indexes], relation_labels[test_indexes]
            
            if verbose > 0:
                print('\n\tTraining K-NN for new kfold configuration...')

            knn.fit(train_data, train_labels, features_types)

            if verbose > 0:
                print('\tClassifying data_test...')
 
            test_data_len = len(test_data)
            pred_labels = []
            for index, features in enumerate(test_data):
                if verbose > 1:
                    if index != 0 and index % int(test_data_len/10) == 0:
                        percentage = (index/len(test_data))*100
                        print('\t\tClassified ' 
                                    + '{:5.2f}'.format(percentage) + '% of ' 
                                    + relation + ' database')
                        if verbose > 2:
                            print('\t\t\t', classification_report(test_labels[:index-1], pred_labels[:index-1]).replace('\n', '\n\t\t\t'))

                ordered_pred_labels = knn.predict(features)
                pred_label, distances = ordered_pred_labels[0]
                pred_labels.append(pred_label)

            metrics.append(precision_recall_fscore_support(test_labels, pred_labels, average='weighted'))

            if verbose > 0:
                precisions = [precision for precision, _, _, _ in metrics]
                recalls = [recall for _, recall, _, _ in metrics]
                f1s = [f1 for _, _, f1, _ in metrics]
                file_.write('Partial results using k = ' + str(knn.k))
                file_.write('\n\tPartial average precision: ' + str(np.mean(precisions)))
                file_.write('\n\tPartial standard deviation: ' + str(np.std(precisions)))
                file_.write('\n\tPartial average recall: ' + str(np.mean(recalls)))
                file_.write('\n\tPartial recall standard deviation: ' + str(np.std(recalls)))
                file_.write('\n\tPartial average F1-Score: ' + str(np.mean(f1s)))
                file_.write('\n\tPartial F1-Score standard deviation: ' + str(np.std(f1s)) + '\n\n')

                if verbose > 1:
                    print('\t\t', classification_report(test_labels, pred_labels).replace('\n', '\n\t\t'))

        precisions = [precision for precision, _, _, _ in metrics]
        recalls = [recall for _, recall, _, _ in metrics]
        f1s = [f1 for _, _, f1, _ in metrics]
        file_.write('Results using k = ' + str(knn.k))
        file_.write('\n\tAverage precision: ' + str(np.mean(precisions)))
        file_.write('\n\tPrecision standard deviation: ' + str(np.std(precisions)))
        file_.write('\n\tAverage recall: ' + str(np.mean(recalls)))
        file_.write('\n\tRecall standard deviation: ' + str(np.std(recalls)))
        file_.write('\n\tAverage F1-Score: ' + str(np.mean(f1s)))
        file_.write('\n\tF1-Score standard deviation: ' + str(np.std(f1s)))
        file_.write('\n\n=======================\n\n')
