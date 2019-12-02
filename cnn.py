import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt
import numpy as np

class ConvNN:

    def __init__(self, num_epochs=10, metrics_list=['accuracy'], optimizer='adam',loss='sparse_categorical_crossentropy',):
        self.num_epochs = num_epochs
        self.metrics = metrics_list
        self.optimizer = optimizer
        self.loss = loss
    
    def make_model(self, train_images, train_labels):
        self.train_images = train_images
        self.train_labels = train_labels

        self.model = models.Sequential()
        self.model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(np.unique(train_labels).shape[0], activation='softmax'))
        self.model.compile(optimizer=self.optimizer,
              loss=self.loss,
              metrics=self.metrics)

        
    def fit(self, test_images, test_labels):
        self.history = self.model.fit(self.train_images, self.train_labels, epochs=self.num_epochs, 
                    validation_data=(test_images, test_labels))

    def plot_all(self, test_images, test_labels):
        plt.plot(self.history.history['acc'], label="accuracy")
        plt.plot(self.history.history['val_acc'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')

        self.test_loss, self.test_acc = self.model.evaluate(test_images,  test_labels, verbose=2)
        print("Loss: ", self.test_loss, " | Accuracy: ", self.test_acc)