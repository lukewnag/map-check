import geopandas as gpd
# df = gpd.read_file("GA/GA_precincts16.shp")

state = 'OR'

from gerrychain import Graph
import json
df = gpd.read_file(state+"2011/"+state+".shp", ignore_errors=True)

import matplotlib.pyplot as plt
df.plot()
plt.show()
# dual_graph = Graph.from_geodataframe(df)
# dual_graph.to_json(state+"2011/"+state+"_VTDs_init.json")
# dual_graph3 = Graph.from_json("GA/GA_graph.json")


# source for files - districtbuilder!