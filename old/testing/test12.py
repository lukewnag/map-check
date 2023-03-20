from gerrychain import (Graph, GeographicPartition, Election, updaters)
import numpy as np

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

import matplotlib.pyplot as plt
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain, proposals, updaters, constraints, accept, Election)
from gerrychain.proposals import recom
from functools import partial
from pandas import DataFrame



YEAR = "2011"
ELECTION_USED = 'GOV18'
STATE = 'OR'

dem_vote, gop_vote = 0, 0
if ELECTION_USED == "PRES16":
    dem_vote = dem_field_pres16[STATE]
    gop_vote = gop_field_pres16[STATE]
elif ELECTION_USED == "GOV18":
    dem_vote = dem_field_gov18[STATE]
    gop_vote = gop_field_gov18[STATE]
elif ELECTION_USED == "HOUSE18":
    dem_vote = dem_field_house18[STATE]
    gop_vote = gop_field_house18[STATE]
POP_FIELD_NAME = pop_field[STATE]
WHITE_POP = white_pop_field[STATE]
COUNTY_FIELD_NAME = county_field[STATE]
ASSIGNMENT = assignment_field[STATE]

graph = Graph.from_json(''.join([STATE, YEAR, '/', STATE, '_VTDs.json']))

elections = []
elections.append(Election(ELECTION_USED, {"Democratic": dem_vote, "Republican": gop_vote}))

# Population updater, for computing how close to equality the district
# populations are. POP_FIELD_NAME is the population column from our shapefile.
my_updaters = {"population": updaters.Tally(POP_FIELD_NAME, alias="population")}

# Election updaters, for computing election results using the vote totals
# from our shapefile.
election_updaters = {election.name: election for election in elections}
my_updaters.update(election_updaters)

partition = GeographicPartition(graph, assignment=ASSIGNMENT, updaters=my_updaters)
# print(overall_polsby_popper(initial_partition))


import math


def compute_polsby_popper(area, perimeter):
    try:
        return 4 * math.pi * area / perimeter ** 2
    except ZeroDivisionError:
        return math.nan


def polsby_popper(partition):
    """Computes Polsby-Popper compactness scores for each district in the partition.
    """
    return {
        part: compute_polsby_popper(
            partition["area"][part], partition["perimeter"][part]
        )
        for part in partition.parts
    }

print(polsby_popper(partition))
print(partition['area']['02'], partition['perimeter']['02'])