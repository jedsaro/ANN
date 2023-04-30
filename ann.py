import csv
import numpy as np

NODES_PER_LAYER = [4, 2, 1] 

class ANN:
    def __init__(self):
        self.W1 = np.random.uniform(0,1, size =  (NODES_PER_LAYER[0], NODES_PER_LAYER[1]))
        self.W2 = np.random.uniform(0,1, size= (NODES_PER_LAYER[1], NODES_PER_LAYER[2]))

    def feedForward(self, inputs):
        self.layer2 = np.dot(inputs, self.W1)
        self.layer_2_activate = self.sigmoid(self.layer2)
        self.layer3 = np.dot(self.layer_2_activate, self.W2)
        output = self.sigmoid(self.layer3)
        return output
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return np.exp(-x)/((1+np.exp(-x))**2)
    
    def update_weights(self,l_rate, update_layer1, update_layer2):
        self.W1 -= l_rate * update_layer1
        self.W2 -= l_rate * update_layer2
    
    def backpropagate(self, inputs , output, expected):
        delta3 = np.multiply(-(expected-output), self.sigmoid_derivative(self.layer3))
        dJdW2 = np.dot(self.layer_2_activate.T, delta3)
        deltlayer_2_activate = np.dot(delta3, self.W2.T) * self.sigmoid_derivative(self.layer2)
        dJdW1 = np.dot(inputs.T, deltlayer_2_activate)  
        return dJdW1, dJdW2
    
    def training(self,l_rate, n_epoch, target_error):
        for epoch in range(n_epoch):
            sum_error = 0
            with open('index.csv') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    float_row = np.array([float(element) for element in row[:4]]).reshape(1, 4)
                    expected = np.array([float(row[i]) for i in range(4, 5)])
                    output = self.feedForward(float_row)
                    for i in range((NODES_PER_LAYER[-1])): 
                        sum_error += (expected-output)**2
                        dJdW1, dJdW2 = self.backpropagate(float_row, output, expected)
                        self.update_weights(l_rate, dJdW1, dJdW2)

            if sum_error <= target_error:
                print("Target Error Reached error=%.3f" % (sum_error))
                return

            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
                        
if __name__ == '__main__':
    ann = ANN()
    ann.training(.8,1000000000,.5)
