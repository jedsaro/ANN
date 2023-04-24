import csv, random

NODES_PER_LAYER = [4, 2, 1] 
INPUT_VALUES = [4.1, 5.5, 3.3, 10.1]   

class Node:
    def __init__(self, name):
        self.children = []
        self.weight = []
        self.name = name
        self.input = None
        self.information = self.readFile()

    def make_children(self, current_layer, nodes_per_layer, past_sum=0):
        if current_layer >= len(nodes_per_layer):
            return
        
        #creates layers
        for i in range(nodes_per_layer[current_layer]):
            self.children.append(Node((f"layer[{current_layer}]---- Node - {i+1}")))

            if(current_layer == 0):
                self.children[i].input = INPUT_VALUES[i]

            if current_layer > 0:
                self.children[i].input = past_sum

            if i == 0:
                next_sum = self.children[i].input
            else:
                next_sum += self.children[i].input

        #traverses NODE_PER_LAYER on the first node
        self.children[0].make_children(current_layer + 1, nodes_per_layer, next_sum)

        #copies children to other nodes
        for i in range(1, len(self.children)):
            self.children[i].children = self.children[0].children[:]

    def readFile(self):
        with open('index.csv') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                first_four = row[:4]
                return first_four
            
    def random_weights(self, current_layer, node_per_layer):
        if current_layer >= len(node_per_layer):
            return
        
        self.weight = [0.0] * len(self.children)

        for i in range(len(self.children)):

            self.weight[i] = random.uniform(0, 1)

            self.children[i].random_weights(current_layer + 1, node_per_layer)

        return



    def print(self):
        for i in range(len(self.children)):
            try: 

                print(f"Weight of {self.weight[i]}")

            except:
                pass
            
        for i in range(NODES_PER_LAYER[0]):
            print(self.children[i].name)
            print(self.children[i].input)


        print(self.children[0].children[0].name)
        print(self.children[0].children[0].input)

        print(self.children[0].children[1].name)
        print(self.children[0].children[1].input)

        print(self.children[0].children[0].children[0].name)
        print(self.children[0].children[0].children[0].input)
        
            

if __name__ == '__main__':
    node = Node("Master")
    node.make_children(0, NODES_PER_LAYER)
    node.random_weights(0, NODES_PER_LAYER)
    node.print()