import csv
import numpy as np

NODES_PER_LAYER = [4, 2, 1] 

class Node:
    def __init__(self):
        self.W1 = np.random.randn(NODES_PER_LAYER[0], NODES_PER_LAYER[1])
        self.W2 = np.random.randn(NODES_PER_LAYER[1], NODES_PER_LAYER[2])
        self.information = self.readFile()

    def feedFoward(self, input_values):
        self.z2 = np.dot(input_values, self.W1)
        self.a2 = self.sigmoidx(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        output = self.sigmoidx(self.z3) 
        return output
        
    def sigmoidx(self, x):
        return 1 / (1 + np.exp(-x))
    
    def readFile(self):
        with open('index.csv') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                first_four = row[:4]
                return first_four
            
    def print(self):
    #w1s
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