import sqlite3
import numpy as np

NODES_PER_LAYER = [784, 2, 1] 

conn = sqlite3.connect('datas.db')
cursor = conn.cursor()

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
        delta3_activate = np.multiply(-(expected-output), self.sigmoid_derivative(self.layer3))
        dJdW2 = np.dot(self.sigmoid_derivative(self.layer3), delta3_activate)
        delta2_activate = np.dot(delta3_activate, self.W2.T) * self.sigmoid_derivative(self.layer2)
        dJdW1 = np.dot(inputs.T, delta2_activate)  
        return dJdW1, dJdW2
    
    def training(self,l_rate, n_epoch, target_error):
        for epoch in range(n_epoch):
            sum_error = 0
            cursor.execute('select * from e_train order by random() limit 10;')
            train = cursor.fetchall()
            for row in train:
                float_row = np.array([float(element) for element in row[1:]]).reshape(1, NODES_PER_LAYER[0]) 
                expected = np.array([float(row[0])])
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
    ann.training(1,100000000,.1)