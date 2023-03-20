from gerrychain import (Graph, GeographicPartition, Election, updaters)
import numpy as np

STATE = 'TX'

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


# MINORITY OPPORTUNITY:
# I do not know how to do a proper ecological regression. Therefore, I will be using a simplified
#   model with the following (VERY crude) assumptions:
# If district is at least 53% minority: minority opportunity
# Otherwise, if the district votes Democrat and the minority proportion is at least 62.5% of the Democrat
#   vote share: minority opportunity (this assumes all minorities vote 80% Democrat regardless of location)
# The first condition is intended for places where voting is racially polarized, e.g. the South
# The second is intended for places where there is a lot of crossover voting, e.g. diverse urban areas
# We can categorize how strong the minority opportunity is by looking at the probability of a Democrat win
#   and by looking at how strongly minority it is.
# This ignores the possibility that minority voters could favor Republicans, as they do in Hialeah, FL.
#   (It should be noted that Hialeah is covered by Condition #1, however, as Condition #1 is party blind.)
# It also doesn't account for the fact that different minorities have different vote distributions.
# Finally, this only accounts for "coalition" districts - it doesn't have anything for, say, majority-Hispanic.

# Racial packing - maybe have this trigger if the district is over 70% minority?

def minority_opportunity(partition):
    opportunity_districts = set()
    for district in partition.parts:
        tot_pop, minority_pop = 0, 0
        for node in partition.parts[district]:
            tot_pop += partition.graph.nodes[int(node)][POP_FIELD_NAME]
            minority_pop += partition.graph.nodes[int(node)][POP_FIELD_NAME] - partition.graph.nodes[int(node)][WHITE_POP]
        if minority_pop >= 0.53*tot_pop: opportunity_districts.add(district)
        else:
            dem, gop = 0,0
            for node in partition.parts[district]:
                dem += partition.graph.nodes[int(node)][DEM_VOTE]
                gop += partition.graph.nodes[int(node)][GOP_VOTE]
            if dem >= 0.53*(dem+gop) and minority_pop/tot_pop >= 0.625*dem/(dem+gop):
                opportunity_districts.add(district)
    return opportunity_districts