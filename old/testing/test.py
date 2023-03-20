import networkx as nx
import random

graph = nx.grid_graph([20,20])
for e in graph.edges():
    graph[e[0]][e[1]]["shared_perim"] = 1
for n in graph.nodes():
    graph.nodes[n]["population"] = 1
    if random.random() < .4:
        graph.nodes[n]["pink"] = 1
        graph.nodes[n]["purple"] = 0
    else:
        graph.nodes[n]["pink"] = 0
        graph.nodes[n]["purple"] = 1
    if 0 in n or 19 in n:
        graph.nodes[n]['boundary_node'] = True
        graph.nodes[n]['boundary_perim'] = 1
    else:
        graph.nodes[n]['boundary_node'] = False

# print(graph.nodes(data="pink"))
assignment_dict = {x: int(x[0]/10) for x in graph.nodes()}
#values = [assignment_dict[x] for x in graph.nodes()]
values = [graph.nodes(data="pink")[x] for x in graph.nodes()]

import matplotlib.pyplot as plt
nx.draw(graph, node_color = values)
plt.show()