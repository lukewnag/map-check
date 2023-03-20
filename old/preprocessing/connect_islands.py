from gerrychain import Graph

state = 'OH'
graph = Graph.from_json(f"{state}2011/{state}_VTDs.json")

def find_node_by_geoid(geoid, graph=graph):
    for node in graph:
        if graph.nodes[node]["PRECODE"] == geoid:
            return node

connections = open(f'{state}2011/{state}_connections.txt', 'r').read().splitlines()
for pair in connections:
    node1, node2 = pair.split(', ')
    graph.add_edge(find_node_by_geoid(node1), find_node_by_geoid(node2))

# islands = graph.islands
# print(islands)

# from networkx import connected_components, is_connected
# components = list(connected_components(graph))
# print([len(c) for c in components])

# print(is_connected(graph))

graph.to_json(f"{state}2011/temp.json")