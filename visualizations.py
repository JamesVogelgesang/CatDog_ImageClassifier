import matplotlib.pyplot as plt
import json

# Graph: Val Loss vs Loss
def visual1():
    with open('historyData.json', 'r') as file:
        loadedHistoryjson = file.read()

    loadedHistory = json.loads(loadedHistoryjson)

    # Visualization #1, graphs Loss vs Validation Loss of test data during training
    fig = plt.figure('Figure 1')
    plt.plot(loadedHistory['loss'], color='red', label='Loss')
    plt.plot(loadedHistory['val_loss'], color='blue', label='Val Loss')
    fig.suptitle('Loss During Validation Vs Loss During Training', fontsize=20)
    plt.legend(loc='upper right')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    
    plt.show()

# Graph: Accuracy Over 20 Epochs
def visual2():
    with open('historyData.json', 'r') as file:
            loadedHistoryjson = file.read()

    loadedHistory = json.loads(loadedHistoryjson)

    # Visualization #2, graphs the accuracy of each epoch during training
    fig = plt.figure('Figure 2')
    plt.plot(loadedHistory['accuracy'], color='black')
    fig.suptitle('Accuracy Through Out Training', fontsize=20)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    
    plt.show()

# Bar plot: Acc, Pre, Recall Against Test Data
def visual3():
    with open('testData.json', 'r') as file:
        loadedTestData = json.load(file)

    # Visualization #3, shows final accuracy, recall, and precision of training data in a bar chart
    fig1 = plt.figure('Figure 3')
    x_axis = ['Accuracy', 'Recall', 'Precision']
    y_axis = [loadedTestData['accuracy'][1], loadedTestData['recall'][1], loadedTestData['precision'][1]]
    colors = ['tab:red', 'tab:blue', 'tab:orange']
    barplot = plt.bar(x=x_axis, height=y_axis, color=colors)
    plt.bar_label(barplot, labels=y_axis, label_type='center')
    fig1.suptitle('Final Accuracy, Recall, and Precision of Test Data')
    
    plt.show()

#visual1()
#visual2()
#visual3()

