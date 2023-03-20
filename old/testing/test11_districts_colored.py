from gerrychain import (Graph, GeographicPartition, Election, updaters)
import numpy as np

STATE = 'PA'

YEAR = "2011"

ELECTION_USED = "PRES16"

natlDemVote = {"PRES12": 51.964, "PRES16": 51.113, "HOUSE18": 54.379, "PRES20": 52.294}

# some states do not have 2016 results
# NC has multiple assignments: oldplan (2011), newplan (2016), judge (2017)
dem_field_pres16 = {'GA': 'PRES16D', 'MI': 'PRES16D', 'MN': 'PRES16D', 'NC': 'EL16G_PR_D', 'OH': 'PRES16D',
                    'OR': 'PRES16D', 'PA': 'T16PRESD', 'TX': 'PRES16D', 'VA': 'G16DPRS', 'WI': 'PREDEM16'}
gop_field_pres16 = {'GA': 'PRES16R', 'MI': 'PRES16R', 'MN': 'PRES16R', 'NC': 'EL16G_PR_R', 'OH': 'PRES16R',
                    'OR': 'PRES16R', 'PA': 'T16PRESR', 'TX': 'PRES16R', 'VA': 'G16RPRS', 'WI': 'PREREP16'}
dem_field_gov18 = {'AZ': 'GOV18D', 'CO': 'GOV18D', 'MI': 'GOV18D', 'OR': 'GOV18D'}
gop_field_gov18 = {'AZ': 'GOV18R', 'CO': 'GOV18R', 'MI': 'GOV18R', 'OR': 'GOV18R'}
dem_field_house18 = {'AZ': 'USH18D', 'CO': 'USH18D', 'OR': 'USH18D'}
gop_field_house18 = {'AZ': 'USH18R', 'CO': 'USH18R', 'OR': 'USH18R'}
pop_field = {'AZ': 'TOTPOP', 'CO': 'TOTPOP', 'GA': 'TOTPOP', 'MI': 'TOTPOP', 'MN': 'TOTPOP', 'NC': 'TOTPOP',
             'OH': 'TOTPOP', 'OR': 'TOTPOP', 'PA': 'TOTPOP', 'TX': 'TOTPOP', 'VA': 'TOTPOP', 'WI': 'PERSONS'}
assignment_field = {'AZ': 'CD', 'CO': 'CD116FP', 'GA': 'CD', 'MI': 'CD', 'MN': 'CONGDIST', 'NC': 'newplan',
                    'OH': 'CD', 'OR': 'CD', 'PA': 'CD_2011', 'TX': 'USCD', 'VA': 'CD_16', 'WI': 'CON'}
projection_code = {'AZ': '2223', 'CO': '2957', 'GA': '4019', 'MI': '6493', 'MN': '26915', 'NC': '6543',
                   'OH': '3747', 'OR': '2992', 'PA': '26918', 'TX': '3081', 'VA': '3968', 'WI': '26916'}
county_field = {'AZ': 'COUNTY', 'CO': 'COUNTYFP', 'GA': 'CTYNAME', 'MI': 'county_nam', 'MN': 'COUNTYNAME', 'NC': 'County',
                'OH': 'COUNTY', 'OR': 'County', 'PA': 'COUNTYFP10', 'TX': 'COUNTY', 'VA': 'locality', 'WI': 'CNTY_NAME'}
white_pop_field = {'AZ': 'NH_WHITE', 'CO': 'NH_WHITE', 'GA': 'NH_WHITE', 'MI': 'NH_WHITE', 'MN': 'NH_WHITE', 'NC': 'NH_WHITE',
             'OH': 'NH_WHITE', 'OR': 'NH_WHITE', 'PA': 'NH_WHITE', 'TX': 'WHITE', 'VA': 'NH_WHITE', 'WI': 'WHITE'}

DEM_VOTE, GOP_VOTE = 0, 0
if ELECTION_USED == "PRES16":
    DEM_VOTE = dem_field_pres16[STATE]
    GOP_VOTE = gop_field_pres16[STATE]
elif ELECTION_USED == "GOV18":
    DEM_VOTE = dem_field_gov18[STATE]
    GOP_VOTE = gop_field_gov18[STATE]
elif ELECTION_USED == "HOUSE18":
    DEM_VOTE = dem_field_house18[STATE]
    GOP_VOTE = gop_field_house18[STATE]
POP_FIELD_NAME = pop_field[STATE]
COUNTY_FIELD_NAME = county_field[STATE]
ASSIGNMENT = assignment_field[STATE]
WHITE_POP = white_pop_field[STATE]

graph = Graph.from_json(''.join([STATE, YEAR, '/', STATE, '_VTDs.json']))

elections = []
elections.append(Election(ELECTION_USED, {"Democratic": DEM_VOTE, "Republican": GOP_VOTE}))

# Population updater, for computing how close to equality the district
# populations are. POP_FIELD_NAME is the population column from our shapefile.
my_updaters = {"population": updaters.Tally(POP_FIELD_NAME, alias="population")}

# Election updaters, for computing election results using the vote totals
# from our shapefile.
election_updaters = {election.name: election for election in elections}
my_updaters.update(election_updaters)

partition = GeographicPartition(graph, assignment=ASSIGNMENT, updaters=my_updaters)


# MAP IT!

from matplotlib.colors import LinearSegmentedColormap
def minority_heatmap(partition):
    # heatmap_colors = [(0, (0,0,0)), (0.5, (0.9,0.9,0.9)), (1, (0.5,0.7,0))]
    minority_proportions = {}
    for district in partition.parts:
        tot_pop, minority_pop = 0, 0
        for node in partition.parts[district]:
            tot_pop += partition.graph.nodes[int(node)][POP_FIELD_NAME]
            minority_pop += partition.graph.nodes[int(node)][POP_FIELD_NAME] - partition.graph.nodes[int(node)][WHITE_POP]
        minority_proportions[district] = minority_pop/tot_pop
    
    district_colors = [0 for i in range(len(partition.parts))]
    districts = [int(district) for district in partition.parts]
    for district in partition.parts:
        if 0 in districts: district_num = int(district)
        else: district_num = int(district)-1
        minority_percent = minority_proportions[district]
        if minority_percent < 0.5:
            gray = 1.8*minority_percent
            district_colors[district_num] = (district_num/(len(partition.parts)-1), (gray, gray, gray))
        else:
            minority_percent -= 0.5
            red = 0.9 - 0.8*minority_percent
            green = 0.9 - 0.4*minority_percent
            blue = 0.9 - 1.8*minority_percent
            district_colors[district_num] = (district_num/(len(partition.parts)-1), (red, green, blue))

    return LinearSegmentedColormap.from_list("custom", district_colors)

def partisan_heatmap(partition):
    # heatmap_colors = [(0, (0.8,0,0)), (0.5, (0.8,0.8,0.8)), (1, (0,0,0.8))]
    dem_proportions = {}
    for district in partition.parts:
        dem_vote, gop_vote = 0, 0
        for node in partition.parts[district]:
            dem_vote += partition.graph.nodes[int(node)][DEM_VOTE]
            gop_vote += partition.graph.nodes[int(node)][GOP_VOTE]
        dem_proportions[district] = dem_vote/(dem_vote+gop_vote)
    
    district_colors = [0 for i in range(len(partition.parts))]
    districts = [int(district) for district in partition.parts]
    for district in partition.parts:
        if 0 in districts: district_num = int(district)
        else: district_num = int(district)-1
        dem_pct = dem_proportions[district]
        if dem_pct < 0.4:
            district_colors[district_num] = (district_num/(len(partition.parts)-1), (0.2+1.5*dem_pct, 0, 0))
        elif dem_pct > 0.6:
            dem_pct = 1 - dem_pct #flip it around
            district_colors[district_num] = (district_num/(len(partition.parts)-1), (0, 0, 0.2+1.5*dem_pct))
        elif dem_pct < 0.5:
            shade = 8*dem_pct-3.2
            district_colors[district_num] = (district_num/(len(partition.parts)-1), (0.8, shade, shade))
        else:
            dem_pct = 1 - dem_pct #flip it around
            shade = 8*dem_pct-3.2
            district_colors[district_num] = (district_num/(len(partition.parts)-1), (shade, shade, 0.8))

    return LinearSegmentedColormap.from_list("custom", district_colors)

import geopandas
import matplotlib.pyplot as plt
units = geopandas.read_file(''.join([STATE, YEAR, '/', STATE, '.shp']))
units.to_crs({"init": "epsg:"+projection_code[STATE]}, inplace=True)

partition.plot(units, figsize=(10, 7), cmap=minority_heatmap(partition)) # range color, color scale
plt.axis('off')

partition.plot(units, figsize=(10, 7), cmap=partisan_heatmap(partition)) # range color, color scale
plt.axis('off')

plt.show()

# OTHER CODE

# MINORITY PROPORTION OF EACH NODE
# minority_to_unit_dict = {}
# for district in partition.parts:
#     for node in partition.parts[district]:
#         if partition.graph.nodes[node][POP_FIELD_NAME] == 0: minority_to_unit_dict[node] = 0.5
#         else: minority_to_unit_dict[node] = 1 - partition.graph.nodes[node][WHITE_POP]/partition.graph.nodes[node][POP_FIELD_NAME]
# node_minority_proportions = [minority_to_unit_dict[i] for i in range(len(minority_to_unit_dict))]
