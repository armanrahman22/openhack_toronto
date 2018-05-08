from __future__ import print_function

import argparse
import glob
import os

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D, BatchNormalization
from keras.models import Sequential
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from skimage import img_as_float, io
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

labels = ['crampons', 'tents', 'boots', 'gloves', 'pulleys', 'carabiners', 
            'axes', 'harnesses', 'hardshell_jackets', 'rope', 'helmets', 'insulated_jackets']

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(labels)


def process_images(main_folder):
    x = np.load(os.path.join(main_folder, "x_image_arrays.npy"))
    y = np.load(os.path.join(main_folder, "y_image_labels.npy"))
    num_classes = len(np.unique(y))
    x /= 255
    y_enc = label_encoder.transform(y)
    y_cat = to_categorical(y_enc)
    input_shape = x[0].shape
    print(input_shape)
    return x, y_cat, input_shape, num_classes


def get_train_test(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=13)
    return x_train, x_test, y_train, y_test


def createModel(input_shape, num_classes, kernel_size):
    model = Sequential()
    
    model.add(Conv2D(filters = 32, kernel_size = (kernel_size, kernel_size), activation='relu', input_shape = input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 32, kernel_size = (kernel_size, kernel_size), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(strides=(2,2)))
    model.add(Dropout(0.10))

    model.add(Conv2D(filters = 32, kernel_size = (kernel_size, kernel_size), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters = 32, kernel_size = (kernel_size, kernel_size), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(strides=(2,2)))
    model.add(Dropout(0.10))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def trainModel(input_shape, num_classes, batch_size, epochs, x_train, x_test, 
               y_train, y_test, train_generator, kernel_size):
    
    model = createModel(input_shape, num_classes, kernel_size)
    model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=1e-4), metrics=["accuracy"])
    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.9 ** x)
    
    history = model.fit_generator(
                train_generator,
                steps_per_epoch= x_train.shape[0]  // batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                callbacks=[annealer])
    
    score = model.evaluate(x_test, y_test, verbose=1)
    print("Score:" + str(score))
    return history, model

def generate_images(x_train, x_test, y_train, y_test, batch_size):
    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    
    train_generator = train_datagen.flow(x_train, y_train, batch_size=batch_size)
    return train_generator 


def plot_loss(history):
    plt.figure(figsize=[8,6])
    plt.plot(history.history['loss'],'r',linewidth=3.0)
    plt.plot(history.history['val_loss'],'b',linewidth=3.0)
    plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Loss',fontsize=16)
    plt.title('Loss Curves',fontsize=16)
    plt.show()


def plot_accuracy(history):
    plt.figure(figsize=[8,6])
    plt.plot(history.history['acc'],'r',linewidth=3.0)
    plt.plot(history.history['val_acc'],'b',linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
    plt.xlabel('Epochs ',fontsize=16)
    plt.ylabel('Accuracy',fontsize=16)
    plt.title('Accuracy Curves',fontsize=16)
    plt.show()


def main(main_folder, batch_size, epochs, kernel_size):
    print("Model batch_size: " + str(batch_size) + " Epochs: " + str(epochs) + " Kernel Size: " + str(kernel_size))
    x, y, input_shape, num_classes = process_images(main_folder)
    x_train, x_test, y_train, y_test = get_train_test(x, y)
    
    train_generator = generate_images(x_train, x_test, y_train, y_test, batch_size)
    history, model = trainModel(input_shape, num_classes, batch_size, epochs, x_train, x_test, 
                       y_train, y_test, train_generator, kernel_size)
    plot_loss(history)
    plot_accuracy(history)
    filepath = main_folder + "/model_2_" + str(batch_size) + "_" + str(epochs) + ".h5"
    model.save(filepath)


if __name__ == "__main__":
    print(K.tensorflow_backend._get_available_gpus())
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True, help="path to input dataset")
    parser.add_argument("-b", "--batch_size", required=True, help="Batch Size")
    parser.add_argument("-e", "--epochs", required=True, help="Epochs")
    parser.add_argument("-k", "--kernel_size", required=True, help="Kernel Size")
    args = parser.parse_args()
    if args.dataset and args.batch_size and args.epochs:
        main(args.dataset, int(args.batch_size), int(args.epochs), int(args.kernel_size))
