import networkx as nx

# create a dynamic graph, add edges and update edge weights
class TransGraph:
    def __init__(self, graph=None, graph_type = "undirected"):
        self.node_count = 512
        if graph:
            self.graph = graph
        elif graph_type == "undirected":
            self.graph = nx.Graph()
        else:
            self.graph = nx.DiGraph()
        self.graph_type = graph_type

    def bestNeighbor(self, node):
        return min(self.graph.neighbors(node), key=lambda neighbor: self.graph[node][neighbor]['weight'])

    def shortest_path(self, start_node, end_node):
        path = nx.dijkstra_path(self.graph, source=start_node, target=end_node)
        print("--------------------------------------------------------------")
        print(f"Path from {start_node} to {end_node}: {path}")
        return path

class MoGraph:
    def __init__(self, graph=None, graph_type = "directed"):
        self.node_count = 512
        if graph:
            self.graph = graph
        elif graph_type == "undirected":
            self.graph = nx.Graph()
        else:
            self.graph = nx.MultiDiGraph()
        self.graph_type = graph_type


    