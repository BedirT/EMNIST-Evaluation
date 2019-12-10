import argparse

import numpy as np
import emnist
import matplotlib.pyplot as plt

import cnn

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.decomposition import PCA

# https://pypi.org/project/emnist/

def load_data(datasetName):
    return emnist.extract_training_samples(datasetName), emnist.extract_test_samples(datasetName)

def visualize_data_grid(imgs, labels):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(imgs[i])
        plt.xlabel(labels[i])
    plt.show()

def reshape_bw(data):
    sh = data.shape
    return data.reshape((sh[0], sh[1], sh[2], 1))

def reshape_flatten(data):
  return data.reshape([data.shape[0], data.shape[1] * data.shape[2]])

if __name__ == "__main__":

    f1=open('./results_naive.txt', 'w+')
    # for ds_name in emnist.list_datasets():
    f1.write('-'*20 + '\nRunning on the dataset '+ emnist.list_datasets()[0] + '\n' + '-'*20 + '\n')
    print('-'*20 + '\nRunning on the dataset '+ emnist.list_datasets()[0] + '\n' + '-'*20)

    (train_images, train_labels), (test_images, test_labels) = load_data(emnist.list_datasets()[0])

    train_images = reshape_bw(train_images)
    test_images = reshape_bw(test_images)
    train_images, test_images = train_images / 255.0, test_images / 255.0

    # CNN = cnn.ConvNN()
    # CNN.make_model(train_images, train_labels)
    # CNN.fit(test_images, test_labels)
    # CNN.plot_all(test_images, test_labels)

    n = 1000

    train_images = reshape_flatten(train_images)#[:n]
    test_images = reshape_flatten(test_images)#[:n]

    # train_labels = train_labels[:n]
    # test_labels = test_labels[:n]

    algos = {
        'Naive Bayes': GaussianNB(),
        # 'Logistic Regression': LogisticRegression(verbose=True),
        # 'KNN': KNeighborsClassifier(n_neighbors = 60, n_jobs = -1),
        # 'Random Forest': RandomForestClassifier(n_estimators=100),
        # 'SVM': svm.SVC(gamma='scale', C=10, kernel='linear', verbose=True),
    }
    sfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=2)
    kfold = KFold(n_splits=10, random_state=7)

    fold_types = {'Stratified Cross Validation':sfold, 'Non-Stratified Cross Validation':kfold}

    pca = PCA(n_components=16*16)
    train_images = pca.fit_transform(train_images)
    test_images = pca.transform(test_images)

    for name, fold_type in fold_types.items():
        f1.write('\nRunning with ' + name + '\n')
        for key, clf in algos.items():
            f1.write('\nRunning algorithm: ' + key + '\n')
            clf.fit(train_images, train_labels)
            
            y_predicted = clf.predict(test_images)

            result = cross_validate(clf, test_images, test_labels, cv=fold_type, scoring=['f1_macro', 'accuracy', 'balanced_accuracy'])
            
            f1.write("Accuracy = {} % \n".format(accuracy_score(test_labels, y_predicted)*100))
            f1.write("Classification Report \n {} \n".format(classification_report(test_labels, y_predicted, labels=np.unique(test_labels))))

            for key, data in enumerate(result):
                f1.write('--'*20 + '\n' + data + '\n' + '--'*20 + '\n')
                for j, item in enumerate(result[data]):
                    f1.write(str(j)+ ' -- ' + str(item) + '\n')


    f1.close()