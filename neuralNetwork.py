import tensorflow as tf
import keras
import os
import numpy as np
import json

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten
from keras.metrics import Precision, Recall, BinaryAccuracy

def trainAndSaveNN():
    data = keras.utils.image_dataset_from_directory('data')

    dataIterator = data.as_numpy_iterator()
    batch = dataIterator.next()

    # Cat = 1 Dog = 0

    dataScaled = data.map(lambda x,y: (x/255, y))

    scaledIterator = dataScaled.as_numpy_iterator()

    batch = scaledIterator.next()

    # partition data into training (70%), validating (20%), and testing (10%)
    trainSize = int(len(dataScaled)*0.7)
    validationSize = int(len(dataScaled)*0.2)
    testSize = int(len(dataScaled)*0.1)

    train = dataScaled.take(trainSize)
    validation = dataScaled.skip(trainSize).take(validationSize)
    test = dataScaled.skip(trainSize+validationSize).take(testSize)

    # Create and add layers to the model
    model = Sequential()

    model.add(Conv2D(24, (3,3), 1, activation='relu', input_shape=(256,256,3)))
    model.add(MaxPool2D())

    model.add(Conv2D(32, (3,3), 1, activation='relu'))
    model.add(MaxPool2D())

    model.add(Conv2D(48, (3,3), 1, activation='relu'))
    model.add(MaxPool2D())

    
    model.add(Conv2D(24, (3,3), 1, activation='relu'))
    model.add(MaxPool2D())

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the layers
    model.compile('adam', loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

    # Store data in logs to use in visualization #3 use: tensorboard --logdir=logs
    logdir='logs'
    tensorboardCallback = keras.callbacks.TensorBoard(log_dir=logdir)

    # Train model
    history = model.fit(train, epochs=20, validation_data=validation, callbacks=[tensorboardCallback])

    historyDataString = json.dumps(history.history)

    with open("historyData.json", 'w') as file:
        file.write(historyDataString)

    precision = Precision()
    recall = Recall()
    accuracy = BinaryAccuracy()

    precisionData = []
    recallData = []
    accuracyData = []

    for batch in test.as_numpy_iterator():
        X, y = batch
        change = model.predict(X)

        precision.update_state(y, change)
        recall.update_state(y, change)
        accuracy.update_state(y, change)

        precisionData.append(float(precision.result()))
        recallData.append(float(recall.result()))
        accuracyData.append(float(accuracy.result()))


    metricsData = {
        'precision' : precisionData,
        'recall' : recallData,
        'accuracy' : accuracyData
    }

    with open("testData.json", 'w') as file:
        json.dump(metricsData, file)

    model.save(os.path.join('model','model.h5'))

# call function that trains/saves NN
# trainAndSaveNN()
