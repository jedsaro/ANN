NODES_PER_LAYER = [4, 2, 1] 
INPUT_VALUES = [4.1, 5.5, 3.3, 10.1]   

class Node:
    def __init__(self, name):
        self.children = []
        self.name = name
        self.collector = []
        self.input = None

    def make_children(self, current_layer, nodes_per_layer, past_sum=0):
        if current_layer >= len(nodes_per_layer):
            return
        
        #creates layers
        for i in range(nodes_per_layer[current_layer]):
            self.children.append(Node((f"layer[{current_layer}]---- Node - {i}")))

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


    def print(self):
        print(self.children[0].children[0].input)
        print(self.children[0].children[1].input)
        print(self.children[0].children[1].children[0].input)
        
            

if __name__ == '__main__':
    node = Node("Master")
    node.make_children(0, NODES_PER_LAYER)
    node.print()