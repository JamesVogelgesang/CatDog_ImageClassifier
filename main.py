import tkinter as tk
import tensorflow as tf
import visualizations
import cv2
import numpy as np
import os

from keras.models import load_model
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox

def loadNN(filePath):
    # Loads neural network, passes file to be read, resizes image
    newModel = load_model(os.path.join('model', 'model.h5'))
    img = cv2.imread(filePath)
    resize = tf.image.resize(img, (256, 256))

    # Predicts if image is dog or cat
    np.expand_dims(resize, 0)
    output = newModel.predict(np.expand_dims(resize/255, 0))

    # Displays the prediction's output and accuracy
    return output


def main():
    root = tk.Tk()
    root.geometry("800x800")
    root.title("Cat or Dog")

    label = tk.Label(root, text="Welcome to Cat or Dog!", font=('Arial', 18))
    label.pack(pady=20)   

    # Opens user's files to pick an image, then runs image through the neural network and outputs predictions
    def openImage():
        filePath = filedialog.askopenfilename()  # Open a file dialog to choose an image
        
        if filePath:
            # Open the image using PIL
            pilImage = Image.open(filePath)

            # Validates image format -> only JPEG
            if pilImage.format == 'JPEG':
                # Validates image size to always fit window
                if pilImage.height > (550) or pilImage.width > (550):
                    pilImage = pilImage.resize([550, 550])
                    
                # Convert the PIL image to a Tkinter PhotoImage
                tkImage = ImageTk.PhotoImage(pilImage)

                # Update the label to display the loaded image
                label.config(image=tkImage)
                label.image = tkImage

                output = loadNN(filePath)
                rawOutput = output[0][0]

                if output > 0.5:
                    percentOutput = rawOutput * 100
                    roundedOutput = round(percentOutput, 2)
                    result = f'Result: The picture is a dog\nAccuracy: {roundedOutput}%'
                    predictOutput.config(text=result)
                    predictOutput.text = result
                elif output == 0.5:
                    result = 'FINAL RESULT: It could be either a cat or a dog!'
                    predictOutput.config(text=result)
                    predictOutput.text = result
                else:
                    percentOutput = (1 - rawOutput) * 100
                    roundedOutput = round(percentOutput, 2)
                    result = f'Result: The picture is a cat\nAccuracy: {roundedOutput}%'
                    predictOutput.config(text=result)
                    predictOutput.text = result
                    
            else:
                messagebox.showerror('Wrong Picture Format', 'Only JPEG format is allowed')


    def createVisual1():
        visualizations.visual1()

    def createVisual2():
        visualizations.visual2()

    def createVisual3():
        visualizations.visual3()

    label = tk.Label(root, text="Upload an image for the model to predict!", font=('Arial', 16))
    label.pack(pady=10)

    predictOutput = tk.Label(root, text="", font=('Arial', 16))
    predictOutput.pack()

    # Create a button to open an image
    button = tk.Button(root, text="Upload Image", command=openImage, bg='silver')
    button.pack(pady=5)

    button = tk.Button(root, text="Figure 1", command=createVisual1, bg='silver')
    button.pack(pady=5)

    button = tk.Button(root, text="Figure 2", command=createVisual2, bg='silver')
    button.pack(pady=5)

    button = tk.Button(root, text="Figure 3", command=createVisual3, bg='silver')
    button.pack(pady=5)

    root.mainloop()

if __name__ == "__main__":
    main()