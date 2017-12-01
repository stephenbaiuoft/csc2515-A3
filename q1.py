'''
Question 1 Skeleton Code


'''

import sklearn
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import svm
import matplotlib.pyplot as plt


def load_data():
    # import and filter data
    newsgroups_train = fetch_20newsgroups(subset='train',remove=('headers', 'footers', 'quotes'))
    newsgroups_test = fetch_20newsgroups(subset='test',remove=('headers', 'footers', 'quotes'))

    return newsgroups_train, newsgroups_test


def bow_features(train_data, test_data):
    # Bag-of-words representation
    bow_vectorize = CountVectorizer()
    bow_train = bow_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    bow_test = bow_vectorize.transform(test_data.data)
    feature_names = bow_vectorize.get_feature_names() #converts feature index to the word it represents.
    shape = bow_train.shape
    print('{} train data points.'.format(shape[0]))
    print('{} feature dimension.'.format(shape[1]))
    print('Most common word in training set is "{}"'.format(feature_names[bow_train.sum(axis=0).argmax()]))
    return bow_train, bow_test, feature_names


def tf_idf_features(train_data, test_data):
    # Bag-of-words representation
    tf_idf_vectorize = TfidfVectorizer()
    tf_idf_train = tf_idf_vectorize.fit_transform(train_data.data) #bag-of-word features for training data
    feature_names = tf_idf_vectorize.get_feature_names() #converts feature index to the word it represents.
    tf_idf_test = tf_idf_vectorize.transform(test_data.data)
    return tf_idf_train, tf_idf_test, feature_names


def bnb_baseline(bow_train, train_labels, bow_test, test_labels):
    # training the baseline model
    binary_train = (bow_train>0).astype(int)
    binary_test = (bow_test>0).astype(int)

    model = BernoulliNB()
    model.fit(binary_train, train_labels)

    # evaluate the baseline model
    train_pred = model.predict(binary_train)
    print('BernoulliNB baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = model.predict(binary_test)
    print('BernoulliNB baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return model


# compare cnn model
def dnn_model(bow_train, train_labels, bow_test, test_labels,
              num_layer=5, num_units=100):
    # build 5 layers, of 100 units
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes = (num_layer, num_units), random_state = 1)
    clf.fit(bow_train, train_labels)

    train_pred = clf.predict(bow_train)
    print('dnn baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = clf.predict(bow_test)
    print('dnn baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return clf


# return train and test_pred accuracy
def dnn_model_pred(bow_train, train_labels, bow_test, test_labels,
              num_layer=5, num_units=100):
    # build 5 layers, of 100 units
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes = (num_layer, num_units), random_state = 1)
    clf.fit(bow_train, train_labels)

    train_pred = clf.predict(bow_train)
    #print('dnn baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = clf.predict(bow_test)
    #print('dnn baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return ((train_pred == train_labels).mean(),
            (test_pred == test_labels).mean())



# compute k by k confusion matrix, where c_ij is # of test examples
# that belogns to j, but classised as i
def compute_confusion_matrix(model, test_data, test_labels):
    test_pred = model.predict(test_data)
    # class i from 0-19

    confusion_matrix = []
    for class_j in range(20):
        # get correct_index for each class
        correct_index = np.where(test_labels == class_j)[0]
        col_j =[]
        # get test output given each class i indices
        test_output = test_pred[correct_index]
        # for each of predicted model labels
        for i in range(20):
            # this is # of test_output, labelled as i
            num_i = test_output[test_output == i].shape[0]
            # add to col_j
            col_j.append(num_i)

        # stack them up
        col_j = np.vstack(col_j)
        confusion_matrix.append(col_j)

    # compute the confusion_matrix and return
    confusion_matrix = np.hstack(confusion_matrix)


    modified = confusion_matrix
    # get rid of correct labelled ones
    np.fill_diagonal(modified, 0)

    print("modified_matrix is: \n", modified)

    max_one = np.argmax(modified)
    # print("max_one is: ", max_one)

    max_row = (max_one + 1) // 20
    max_col = max_one % 20
    max_val = modified[max_row][max_col]

    if max_val == np.max(modified):
        print("this currently is the max one")

    print("First\ncorrect class is: ", max_col)
    print("mislablled as: ", max_row)
    print("number of max_val is: ", max_val)

    # now discard this let's find the next max
    modified[max_row][max_col] = -1
    max_one = np.argmax(modified)
    max_row = (max_one + 1)// 20
    max_col = max_one % 20
    max_val = modified[max_row][max_col]

    print("Second\ncorrect class is: ", max_col)
    print("mislablled as: ", max_row)
    print("number of max_val is: ", max_val)

    if max_val == np.max(modified):
        print("this currently is the max one")

    return confusion_matrix


# get the confusion matrix, and also set the dnn_model
def most_confused_classes(bow_train, train_labels, bow_test, test_labels):
    dnn = dnn_model(bow_train, train_labels, bow_test, test_labels, 15)
    compute_confusion_matrix(dnn, bow_test, test_labels)






# compare k-means model
def kmeans_model(bow_train, train_labels, bow_test, test_labels):
    # build 5 layers, of 100 units
    kmeans = KMeans(n_clusters=20, random_state=0, init='k-means++').\
        fit(bow_train, train_labels)

    train_pred = kmeans.predict(bow_train)
    print('kmeans baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = kmeans.predict(bow_test)
    print('kmeans baseline test accuracy = {}'.format((test_pred == test_labels).mean()))

    return kmeans


def knn_model(bow_train, train_labels, bow_test, test_labels):
    k_fold = KFold(n_splits=10)

    # do knn with hyper parameter from k = 1 to k = 11
    k = 1
    opt_pred = 0
    opt_k = 0
    for k in range(1, 10):
        test_pred_sum = 0
        print("k is: ", k, "\n")
        for train_indices, test_indices in k_fold.split(bow_train):
            neigh = KNeighborsClassifier(n_neighbors=k)
            neigh.fit(bow_train[train_indices], train_labels[train_indices])
            train_pred = neigh.predict(bow_train[train_indices])

            train_accuracy = (train_pred == train_labels[train_indices]).mean()
            print('cross validation train accuracy = {}'.format(train_accuracy))
            test_pred = neigh.predict(bow_train[test_indices])

            test_accuracy = (test_pred == train_labels[test_indices]).mean()
            print('cross validation test accuracy = {}'.format(test_accuracy))

            test_pred_sum += test_accuracy
        # get opt_k
        if test_pred_sum > opt_pred:
            opt_pred = test_pred_sum
            opt_k = k

    print("\n\nopt k is:", opt_k)
    neigh = KNeighborsClassifier(n_neighbors=opt_k)
    neigh.fit(bow_train, train_labels)
    train_pred = neigh.predict(bow_train)
    print('opt train accuracy = {}'.format((train_pred == train_labels).mean()))

    test_pred = neigh.predict(bow_test)
    print('opt test accuracy = {}'.format((test_pred == test_labels).mean()))


def svm_model(bow_train, train_labels, bow_test, test_labels):
    svm_model = svm.SVC()
    svm_model.fit(bow_train, train_labels)

    train_pred = svm_model.predict(bow_train)
    print('svm baseline train accuracy = {}'.format((train_pred == train_labels).mean()))
    test_pred = svm_model.predict(bow_test)
    print('svm baseline test accuracy = {}'.format((test_pred == test_labels).mean()))


# prints the test accuracy for different knn parameters
def dnn_hyperparameter(bow_train, train_labels, bow_test, test_labels):
    train_set = []
    test_set = []
    for l in range(3,21):
        train_accur, test_accur = dnn_model_pred(bow_train, train_labels, bow_test, test_labels,
              num_layer=l, num_units=100)
        train_set.append(train_accur)
        test_set.append(test_accur)

    # with offset
    i_max = np.argmax(test_set) + 3
    print("layer with highest test_accuracy is: ", i_max)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(range(3,21), train_set, 'bs', range(3,21), test_set, 'g^')
    plt.ylabel('test accuracy vs num_layers')


    train_set_i = []
    test_set_i = []
    for n in np.linspace(100, 500, 5):
        train_accur, test_accur = dnn_model_pred(bow_train, train_labels, bow_test, test_labels,
              num_layer=int(i_max), num_units=int(n))
        train_set_i.append(train_accur)
        test_set_i.append(test_accur)

    # show the num_units effect
    plt.subplot(212)
    plt.plot(np.linspace(100, 500, 5), train_set_i, 'bs', np.linspace(100, 500, 5), test_set_i, 'g^')
    plt.ylabel('test accuracy vs num_units')
    plt.show()




if __name__ == '__main__':
    train_data, test_data = load_data()
    train_bow, test_bow, feature_names = bow_features(train_data, test_data)

    # well interesting
    idf_train, idf_test, idf_feature_names = tf_idf_features(train_data, test_data)

    # knn model
    # knn = knn_model(train_bow, train_data.target, test_bow, test_data.target)

    # bnb_model = bnb_baseline(train_bow, train_data.target, test_bow, test_data.target)

    # kmeans model
    #kmeans = kmeans_model(train_bow, train_data.target, test_bow, test_data.target)


    # svm model
    #svm_output = svm_model(train_bow, train_data.target, test_bow, test_data.target)

    #print("\nrunning on idf data")
    #svm_output2 = svm_model(idf_train, idf_train.target, idf_test, test_data.target)

    # the confusion_matrix
    # most_confused_classes(train_bow, train_data.target, test_bow, test_data.target)

    # compute hyper paramter for DNN
    dnn_hyperparameter(train_bow, train_data.target, test_bow, test_data.target)