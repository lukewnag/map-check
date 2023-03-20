from gerrychain import Graph

state = 'OH'
graph = Graph.from_json(f"{state}2011/{state}_VTDs.json")

def find_node_by_geoid(geoid, graph=graph):
    for node in graph:
        if graph.nodes[node]["PRECODE"] == geoid:
            return node

print(graph.nodes[2429]["PRECODE"])
for edge in graph.edges:
    if edge[1]==2429: print(graph.nodes[edge[0]]["PRECODE"])
    if edge[0]==2429: print(graph.nodes[edge[1]]["PRECODE"])