import numpy as np
import matplotlib.pyplot as plt

def generateData():
    l = [0, 1, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0,
         0, 1, 1, 1, 1, 0]

    a = [0, 0, 1, 1, 0, 0,
         0, 1, 0, 0, 1, 0,
         0, 1, 1, 1, 1, 0,
         0, 1, 0, 0, 1, 0,
         0, 1, 0, 0, 1, 0]

    b = [0, 1, 1, 1, 0, 0,
         0, 1, 0, 0, 1, 0,
         0, 1, 1, 1, 0, 0,
         0, 1, 0, 0, 1, 0,
         0, 1, 1, 1, 0, 0]

    c = [0, 0, 1, 1, 1, 0,
         0, 1, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0,
         0, 0, 1, 1, 1, 0]

    v = [1, 0, 0, 0, 0, 1,
         1, 0, 0, 0, 0, 1,
         0, 1, 0, 0, 1, 0,
         0, 1, 0, 0, 1, 0,
         0, 0, 1, 1, 0, 0]

    dataset = [l, a, b, c, v]
    for i, letter in enumerate(dataset):
        plt.subplot(1, 5, i + 1)
        plt.imshow(np.array(letter).reshape(5, 6))
    plt.show()

    inputs = [np.array(letter).reshape(1, 30) for letter in dataset]
    return inputs

def generateLabels():
    labels = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ]
    return np.array(labels)

def sigmoidFunc(x):
    return 1 / (1 + np.exp(-x))

def forwardPass(x, w1, w2):
    hidden_input = x.dot(w1)
    hidden_output = sigmoidFunc(hidden_input)

    final_input = hidden_output.dot(w2)
    final_output = sigmoidFunc(final_input)

    return final_output

def initializeWeights(x, y):
    return np.random.randn(x, y).reshape(x, y)

def calculateLoss(pred, true):
    error = np.square(pred - true)
    return np.sum(error) / len(true)

def backwardPass(x, y, w1, w2, learning_rate):
    hidden_input = x.dot(w1)
    hidden_output = sigmoidFunc(hidden_input)

    final_input = hidden_output.dot(w2)
    final_output = sigmoidFunc(final_input)

    output_error = final_output - y
    hidden_error = np.multiply(w2.dot(output_error.T).T, np.multiply(hidden_output, 1 - hidden_output))

    w1_adjustment = x.T.dot(hidden_error)
    w2_adjustment = hidden_output.T.dot(output_error)

    w1 -= learning_rate * w1_adjustment
    w2 -= learning_rate * w2_adjustment

    return w1, w2

def trainNetwork(x, y, w1, w2, learning_rate=0.01, epochs=10):
    accuracies = []
    losses = []
    for epoch in range(epochs):
        epoch_losses = []
        for i in range(len(x)):
            prediction = forwardPass(x[i], w1, w2)
            epoch_losses.append(calculateLoss(prediction, y[i]))
            w1, w2 = backwardPass(x[i], y[i], w1, w2, learning_rate)
        print(f"Epoch: {epoch + 1} ----> Accuracy: {(1 - (sum(epoch_losses) / len(x))) * 100}%")
        accuracies.append((1 - (sum(epoch_losses) / len(x))) * 100)
        losses.append(sum(epoch_losses) / len(x))
    return accuracies, losses, w1, w2

def makePrediction(x, w1, w2):
    output = forwardPass(x, w1, w2)
    predicted_letter = np.argmax(output)
    letters = ["L", "A", "B", "C", "V"]
    print(f"The letter is: {letters[predicted_letter]}")

if __name__ == '__main__':
    x = generateData()
    y = generateLabels()
    print('Training Data:')
    print('x:', x)
    print('y:', y)

    w1 = initializeWeights(30, 5)
    w2 = initializeWeights(5, 5)
    print('Initial Weights:')
    print('w1:', w1)
    print('w2:', w2)

    print('Training the Network:')
    accuracies, losses, w1, w2 = trainNetwork(x, y, w1, w2, 0.1, 400)

    print('Trained Weights:')
    print('w1:', w1)
    print('w2:', w2)

    plt.plot(accuracies)
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.show()

    plt.plot(losses)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.show()

    print("Final output:")
    for i, letter in enumerate(["L", "A", "B", "C", "V"]):
        print(f'Predicting letter "{letter}":')
        makePrediction(x[i], w1, w2)
