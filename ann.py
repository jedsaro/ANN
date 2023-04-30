import csv
import numpy as np

NODES_PER_LAYER = [4, 2, 1] 

class Node:
    def __init__(self):
        #self.data = self.readFile()
        self.inputs = [4.1, 5.5, 3.3, 10.1]
        self.W1 = np.random.randn(NODES_PER_LAYER[0], NODES_PER_LAYER[1])
        self.W2 = np.random.randn(NODES_PER_LAYER[1], NODES_PER_LAYER[2])
        self.collector = []
        self.output = self.feedForward(self.inputs)

    def feedForward(self, input_values):
        self.z2 = np.dot(input_values, self.W1)
        self.a2 = self.sigmoidx(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        output = self.sigmoidx(self.z3) 
        return output
    
    def costFunction(self, x, y):
        return 0.5*sum((y-self.output)**2) 
    
    def sigmoidx(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigPrime(self, x):
        return np.exp(-x)/((1+np.exp(-x))**2)
    
    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:

        delta3 = np.multiply(-(y-self.output), self.sigPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T)*self.sigPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)  
        return dJdW1, dJdW2
    

    
    # def readFile(self):
    #     with open('index.csv') as csvfile:
    #         reader = csv.reader(csvfile)
    #         for row in reader:
    #             first_four = [int(x) for x in row[:4]]
    #             return first_four
            
    def first_train_assign(self, l_rate, n_epoch, target_error):
        number_of_inputs = len(self.inputs)
        for epoch in range(n_epoch):
            sum_error = 0
            with open('index.csv') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    print(row)
                    self.feedForward(row)
                    expected = []
                    for i in range(len(self.network[-1])):
                        expected.append(row[number_of_inputs + i])
                        sum_error += (row[number_of_inputs + i] - self.network[-1][i].collector)**2
                    self.backward_propagate_error(expected)
                    self.update_weights(l_rate)

                if sum_error <= target_error:
                    print("Target Error Reached error=%.3f" % (sum_error))
                    return

                print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

            
    def train_network(self, l_rate, n_epoch, target_error):
        number_of_inputs = len(self.inputs)
        for epoch in range(n_epoch):
            sum_error = 0
            train = self.data
            print(train)


            for row in train:
                print(row)
                self.feedForward(row)
                expected = []
                for i in range(len(self.network[-1])):
                    expected.append(row[number_of_inputs + i])
                    sum_error += (row[number_of_inputs + i] - self.network[-1][i].collector)**2
                self.backward_propagate_error(expected)
                self.update_weights(l_rate)

            if sum_error <= target_error:
                print("Target Error Reached error=%.3f" % (sum_error))
                return

            print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
            
    def print(self):
        for i in range(len(self.W1)):
            try: 
                print(f"{self.W1[i]}")
            except:
                pass

        for i in range(len(self.W2)):
            try: 
                print(f"{self.W2[i]}")
            except:
                pass


if __name__ == '__main__':
    input_values = [4.1, 5.5, 3.3, 10.1]
    ann = Node()
    print(ann.feedFoward(input_values))