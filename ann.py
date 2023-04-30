import csv
import numpy as np

NODES_PER_LAYER = [4, 2, 1] 

class ANN:
    def __init__(self):
        self.inputs = [4.1, 5.5, 3.3, 10.1]
        self.W1 = np.random.randn(NODES_PER_LAYER[0], NODES_PER_LAYER[1])
        self.W2 = np.random.randn(NODES_PER_LAYER[1], NODES_PER_LAYER[2])
        self.collector = []

    def feedForward(self, inputs):
        self.z2 = np.dot(inputs, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        output = self.sigmoid(self.z3) 
        return output
    
    def costFunction(self, X, y):
        return 0.5*sum((y-X)**2) 
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return np.exp(-x)/((1+np.exp(-x))**2)
    
    def backpropagate(self, X, y):
        output_error = y-X
        delta3 = np.multiply(-(output_error), self.sigmoid_derivative(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoid_derivative(self.z2)
        dJdW1 = np.dot(X.T, delta2)  

        return dJdW1, dJdW2
    
    def sum_error(self,x,y):
        return (x - y)**2
        
    
    def first_train_assign(self, n_epoch):
        number_of_inputs = len(self.inputs)
        for epoch in range(n_epoch):
            sum_error = 0
            with open('index.csv') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    float_row = [float(element) for element in row[:4]]
                    output = self.feedForward(float_row)
                    print(output)


if __name__ == '__main__':
    ann = ANN()
    ann.first_train_assign(1)
