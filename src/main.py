from app.retriever.data_collection import DataCollection, EDataType
from app.classifier.knn import KNeighborsClassifier

print('Collecting data...')
data_collection = DataCollection('../res/promise/')

for relation in data_collection.documents:
    print('\nTesting for', relation, 'data set')
    relation_data = data_collection.get_data(relation)
    data_len = len(relation_data)
    print('\tTotal data collected:', data_len)

    # data_split_index = int(data_len/2)
    data_split_index = 15000
    # print('\tData split index:', data_split_index)

    train_data = relation_data[:data_split_index]
    print('\tTotal train_data:', len(train_data))

    test_data = relation_data[data_split_index:]
    print('\tTotal test_data:', len(test_data))

    k = 5
    print('\n\tTraining K-NN where K =', k)
    knn = KNeighborsClassifier(k, train_data, EDataType.NUMERICAL, weighted_distance=True)

    print('\n\tClassifying data_test')
    # test_data = [data for data in test_data if data[1] == '1']
    correct_answers = 0
    incorrect_answers = 0
    for index, data in enumerate(test_data):
        if index != 0 and index % 100 == 0:
            print('\t\tGot into', index, 'from', len(test_data))
            print('\t\t\tPartial correct answers:', correct_answers)
            print('\t\t\tPartial incorrect answers:', incorrect_answers)
            if incorrect_answers == 0:
                incorrect_answers = 1
            print('\t\t\tPartial precision (correct/total):',
                  correct_answers/(correct_answers + incorrect_answers))

        possible_labels, best_match = knn.classify(data[0])
        # print('\n\t\tData:', data[0])
        # print('\t\tTrue label:', data[1])
        # print('\t\tK-NN best label result:', best_match)
        # print('\t\tK-NN all possible label:', possible_labels)

        if best_match == data[1]:
            correct_answers += 1
        else:
            incorrect_answers += 1

    print('\n\tCorrect answers:', correct_answers)
    print('\tIncorrect answers:', incorrect_answers)
    if incorrect_answers == 0:
        incorrect_answers = 1
    print('\tPrecision (correct/total):', correct_answers/(correct_answers + incorrect_answers))    
