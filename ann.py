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
    
    def costFunction(self, X, expected):
        return 0.5*sum((expected-X)**2) 
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return np.exp(-x)/((1+np.exp(-x))**2)
    
    def update_weights(self,l_rate, update_layer1, update_layer2):
        self.W1 -= l_rate * update_layer1
        self.W2 -= l_rate * update_layer2
    
    def backpropagate(self, output, expected):
        delta3 = np.multiply(-(expected-output), self.sigmoid_derivative(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        delta2 = np.dot(delta3, self.W2.T)*self.sigmoid_derivative(self.z2)
        dJdW1 = np.dot(output.T, delta2) 

        return dJdW1, dJdW2

        
    def training(self,l_rate, n_epoch, target_error):
        number_of_inputs = len(self.inputs)
        for epoch in range(n_epoch):
            sum_error = 0
            with open('index.csv') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    float_row = np.array([float(element) for element in row[:4]]).reshape(1, -1)
                    expected = np.array([float(row[i]) for i in range(4, 4 + NODES_PER_LAYER[-1])]).reshape(1, -1)
                    output = self.feedForward(float_row)
                    for i in range(1): 
                        self.collector.append(output)
                        sum_error += (float(row[number_of_inputs + i])- self.collector[-1][i])**2
                        dJdW1, dJdW2 = self.backpropagate(output, expected)
                        self.update_weights(l_rate, dJdW1, dJdW2)
                        #bp_result = self.backpropagate(output[i], expected[-1][i])

            if sum_error <= target_error:
                print("Target Error Reached error=%.3f" % (sum_error))

                return

            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
                        
if __name__ == '__main__':
    ann = ANN()
    ann.training(.8,10,.1)
