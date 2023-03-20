NODES_PER_LAYER = [4, 2, 1] 
INPUT_VALUES = [4.1, 5.5, 3.3, 10.1]   

class Node:
    def __init__(self, name):
        self.children = []
        self.name = name
        self.collector = None

    def make_children(self, current_layer, nodes_per_layer):
        if current_layer >= len(nodes_per_layer):
            return
        
        #creates first layer
        for i in range(nodes_per_layer[current_layer]):
            self.children.append(Node((f"layer[{current_layer}]---- Node - {i}")))
            self.children[i].collector = INPUT_VALUES[i]

        #traverses NODE_PER_LAYER on the first node
        self.children[0].make_children(current_layer + 1, nodes_per_layer)

        #copies children to other nodes
        for i in range(1, len(self.children)):
            self.children[i].children = self.children[0].children[:]

    def print(self, current_layer, nodes_per_layer):

        if current_layer >= len(nodes_per_layer):
            print(self.name)
            return

        print(f"{self.name} is connected to:")

        for i in range(len(self.children)):
            self.children[i].print(current_layer + 1, nodes_per_layer)   
            

if __name__ == '__main__':
    node = Node("Master")
    node.make_children(0, NODES_PER_LAYER)
    node.print(0, NODES_PER_LAYER)