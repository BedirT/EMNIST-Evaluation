import argparse

import numpy as np
import emnist
import matplotlib.pyplot as plt

import cnn

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold, GridSearchCV, cross_val_score

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

    f1=open('./testfile', 'w+')
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

    train_labels = train_labels#[:n]
    test_labels = test_labels#[:n]

    algos = {
        'Naive Bayes': MultinomialNB(),
        # 'Logistic Regression': LogisticRegression(verbose=True),
        # 'KNN': KNeighborsClassifier(),
        # 'Random Forest': RandomForestClassifier(n_estimators=100),
        # 'SVM': svm.SVC(gamma='scale', C=10, kernel='linear', verbose=True),
    }
    
    for key, clf in algos.items():
        print('Running algorithm', key)
        f1.write('\nRunning algorithm' + key + '\n')
        clf.fit(train_images, train_labels)
        
        y_predicted = clf.predict(test_images)

        sfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2)
        kfold = KFold(n_splits=10, random_state=7)
        result = cross_validate(clf, test_images, test_labels, cv=kfold, scoring=['f1_macro', 'accuracy', 'balanced_accuracy'])
        
        # GridSearchCV ---
        
        print("Accuracy = {} %".format(accuracy_score(test_labels, y_predicted)*100))
        # print("Classification Report \n {}".format(classification_report(test_labels, y_predicted, labels=np.unique(test_labels))))    
        # print("Train F1-Values \n {}".format(result['train_f1_macro'])) 
        # f1.write("Train F1-Values \n {}\n".format(result['train_f1_macro']))
        # print("Test F1-Values \n {}".format(result['test_f1_macro']))   
        # f1.write("Test F1-Values \n {}\n".format(result['test_f1_macro'])) 
        # print("Train Accuracy \n {}".format(result['train_accuracy']))   
        # f1.write("Train Accuracy \n {}\n".format(result['train_accuracy'])) 
        # print("Test Accuracy \n {}".format(result['test_accuracy']))   
        # f1.write("Test Accuracy \n {}\n".format(result['test_accuracy'])) 

        for key, data in enumerate(result):
            f1.write('--'*20 + '\n' + data + '\n' + '--'*20 + '\n')
            for j, item in enumerate(result[data]):
                print(j, item)
                f1.write(str(j)+ ' -- ' + str(item) + '\n')


    f1.close()