POP_FIELD_NAME = "TOTPOP"
COUNTY_FIELD_NAME = "COUNTYFP10"
ELECTION_USED = "PRES16"
natlDemVote = {"PRES12": 51.964, "PRES16": 51.113}
TOTAL_STEPS = 600
# WARM_UP_STEPS = 0
STATE = "PA"
ASSIGNMENT = "CD_2011"

import matplotlib.pyplot as plt
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain, proposals, updaters, constraints, accept, Election)
from gerrychain.proposals import recom
from functools import partial
from pandas import DataFrame

#METHODS

# probability of dems winning; stdevs sourced from planscore at https://planscore.org/models/data/2022F/
# this adds about 1 second per 10000 runs (minimal amount of time when paired with lookup table)
from statistics import NormalDist
SAMPLES = 1000 # number of national elections we want to simulate for overall environment
natl_stdev = 3 # how much does the nation swing from election to election?
offsets = [natl_stdev*NormalDist().inv_cdf(i/(SAMPLES+1)) for i in range(1, SAMPLES+1)]
def probDem(natlDemVote, districtDemVote): # natlDemVote is the dem 2 party vote share
    election_stdev = 2 # percent swing in dem vote - district level
    total = 0
    for i in range(SAMPLES):
        total += NormalDist(mu = natlDemVote, sigma = election_stdev).cdf(districtDemVote+offsets[i])
    return total/SAMPLES

# lookup table to speed everything up by a lot - otherwise too repetitive
prob_lookup_table = {50: 0.5}
for i in range(1,1001):
    pct = round(50 + i/100, 2)
    prob_lookup_table[pct] = probDem(50, pct)

def probDemLookup(natlDemVote, districtDemVote):
    expected_vote = districtDemVote + 50 - natlDemVote # adjust for national sentiment
    if expected_vote > 60: return 1
    if expected_vote < 40: return 0
    if expected_vote < 50:
        return 1 - prob_lookup_table[round(100-expected_vote, 2)]
    return prob_lookup_table[round(expected_vote, 2)]

# expected seats in each simulated election
def expectedDem(natlDemVote, dem_voteshare):
    expectedSeats = 0
    for districtVote in dem_voteshare:
        chanceDem = probDemLookup(natlDemVote, 100*districtVote) # convert to percent
        expectedSeats += chanceDem
    return expectedSeats

# compactness - polsby-popper
from gerrychain.metrics import compactness
def overall_polsby_popper(partition):
    polsby_dict = compactness.polsby_popper(partition)
    return sum(polsby_dict[dist] for dist in polsby_dict)/len(polsby_dict)

# this method warns us if there's a district that's really ugly even if the other ones are all compact
def min_polsby_popper(partition):
    polsby_dict = compactness.polsby_popper(partition)
    return min(polsby_dict[dist] for dist in polsby_dict)

# copied from gerrychain.updaters.county_splits
import collections
CountyInfo = collections.namedtuple("CountyInfo", "split nodes contains")
def county_splits(partition, county_field_name):
    """Track nodes in counties and information about their splitting."""

    # Create the initial county data containers.
    county_dict = dict()

    for node in partition.graph:
        county = partition.graph.nodes[node][county_field_name]
        if county in county_dict:
            split, nodes, seen = county_dict[county]
        else:
            split, nodes, seen = 0, [], set()

        nodes.append(node)
        seen.update(set([partition.assignment[node]]))

        if len(seen) > 1:
            split = 1

        county_dict[county] = CountyInfo(split, nodes, seen)

    return county_dict

def num_county_splits(partition, countyFieldName):
    node_counties = (county_splits(partition, countyFieldName))
    counties = node_counties.keys() # contains each county's name - for lookup

    numSplits = 0 # counts number of county splits
    for county in counties:
        if len(node_counties[county][2])>1: numSplits += 1
    return numSplits

# how well does this partition preserve geographic boundaries (county lines)?
# from gerrychain.updaters import county_splits
def geographic_boundary_score(partition, countyFieldName, populationFieldName): #examples: 'COUNTYFP10', 'CNTY_NAME'
    node_counties = (county_splits(partition, countyFieldName))
    counties = node_counties.keys() # contains each county's name - for lookup

    numSplits = 0 # counts number of county splits
    for county in counties:
        if len(node_counties[county][2])>1: numSplits += 1
    score = -numSplits

    node_assignments = partition.assignment # dict with each node's assignment to district
    
    for county in counties:
        if len(node_counties[county][2]) == 1: continue
        districtPop = {} # dictionary to store population of each county in each electoral district
        for node in node_counties[county][1]: # iterates thru each vtd in each county
            if (cd := node_assignments[node]) not in districtPop: districtPop[cd] = 0
            districtPop[cd] += partition.graph.nodes[node][populationFieldName]
        county_cds = districtPop.keys() # electoral districts contained within county
        dist_in_county_pops = [districtPop[dist] for dist in county_cds]
        county_pop = sum(dist_in_county_pops)
        random_guess_correct = max(dist_in_county_pops)/county_pop # chance that a random guess of a resident's district is correct
        score += random_guess_correct

    return score

# each district gets a score of double the weaker party’s probability of winning the seat
# score for entire map is the sum of the scores for each district
# TODO: penalize if the competitives all lean towards one party (>70% chance)
def competitiveness_score(natlDemVote, dem_voteshare):
    score = 0
    for i in range(len(dem_voteshare)):
        dem_pct = dem_voteshare[i]*100 + 50 - natlDemVote #calculate expected vote in D+0 environment
        losing_pct = min(dem_pct, 100-dem_pct)
        score += 2*probDemLookup(natlDemVote, losing_pct) #score can max out at 100
    return 100*score/len(dem_voteshare)

# finds trend line, then uses that to find regression coefficient
# regression of under 0.85 is definitely gerrymandered; regression of 0.9+ is pretty good
# should be compared to ensemble as percentile
import numpy as np
from sklearn.linear_model import LinearRegression
def regression(dem_voteshare):
    x = np.array(sorted(dem_voteshare)).reshape((-1, 1))
    y = np.array(range(len(dem_voteshare)))

    model = LinearRegression()
    model.fit(x, y)
    return model.score(x, y) # returns r coefficient


# MARKOV CHAIN ANALYSIS

graph = Graph.from_json("PA/PA_VTDs.json")

elections = [
    Election("SEN10", {"Democratic": "SEN10D", "Republican": "SEN10R"}),
    Election("SEN12", {"Democratic": "USS12D", "Republican": "USS12R"}),
    Election("SEN16", {"Democratic": "T16SEND", "Republican": "T16SENR"}),
    Election("PRES12", {"Democratic": "PRES12D", "Republican": "PRES12R"}),
    Election("PRES16", {"Democratic": "T16PRESD", "Republican": "T16PRESR"})
]

# Population updater, for computing how close to equality the district
# populations are. "TOTPOP" is the population column from our shapefile.
my_updaters = {"population": updaters.Tally("TOTPOP", alias="population")}

# Election updaters, for computing election results using the vote totals
# from our shapefile.
election_updaters = {election.name: election for election in elections}
my_updaters.update(election_updaters)

initial_partition = GeographicPartition(graph, assignment=ASSIGNMENT, updaters=my_updaters)

# The ReCom proposal needs to know the ideal population for the districts so that
# we can improve speed by bailing early on unbalanced partitions.
ideal_population = sum(initial_partition["population"].values()) / len(initial_partition)

# We use functools.partial to bind the extra parameters (pop_col, pop_target, epsilon, node_repeats)
# of the recom proposal.
proposal = partial(recom,
                   pop_col=POP_FIELD_NAME,
                   pop_target=ideal_population,
                   epsilon=0.02,
                   node_repeats=2
                  )

iterations = 0
shrink_ratio = 2**(-1/100)
shrinking_variable = 2**.5 # tends towards 0 over time; gets multiplied by shrink_ratio every iteration

polsby_goal = 0.18 #this is a fairly low value
def compactness_bound(partition):
    lower_bound = polsby_goal - (polsby_goal - overall_polsby_popper(initial_partition) * 0.9)* shrinking_variable
    return overall_polsby_popper(partition) > lower_bound

pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, 0.02)

# max_splits = 2 + len(initial_partition) # allows for some room for error
# def county_split_bound(partition): ## TODO DELETE
#     upper_bound = max_splits + (num_county_splits(initial_partition, COUNTY_FIELD_NAME) * 1.1 - max_splits)* shrinking_variable
#     return num_county_splits(partition, COUNTY_FIELD_NAME) < upper_bound

geog_limit = -1.8*len(initial_partition) # a good estimate is -1.5* num(districts) - this allows some error
def geog_bound(partition):
    lower_bound = geog_limit + (geographic_boundary_score(initial_partition, COUNTY_FIELD_NAME, POP_FIELD_NAME) * 1.1 - geog_limit)* shrinking_variable
    return geographic_boundary_score(partition, COUNTY_FIELD_NAME, POP_FIELD_NAME) > lower_bound

# competitiveness_bound = constraints.LowerBound(
#     lambda p: round(100*competitiveness_score(natlDemVote[ELECTION_USED], p[ELECTION_USED].percents["Democratic"])),
#     7
# )

chain = MarkovChain(
    proposal=proposal,
    constraints=[
        pop_constraint,
        compactness_bound
        # geog_bound
        # county_split_bound
        # competitiveness_bound
    ],
    accept=accept.always_accept,
    initial_state=initial_partition,
    total_steps=TOTAL_STEPS
)

unprocessed_vote_data = []
avg_polsby_raw, min_polsby_raw = [], []
expected_dems = []
geographic_scores = []
regression_scores = []
competitiveness_scores = []

final_partition = ""

for partition in chain.with_progress_bar():
    dem_voteshare = sorted(partition[ELECTION_USED].percents("Democratic"))
    unprocessed_vote_data.append(dem_voteshare)
    avg_polsby_raw.append(overall_polsby_popper(partition))
    min_polsby_raw.append(min_polsby_popper(partition))
    expected_dems.append(expectedDem(natlDemVote[ELECTION_USED], dem_voteshare))
    geographic_scores.append(geographic_boundary_score(partition, COUNTY_FIELD_NAME, POP_FIELD_NAME))
    regression_scores.append(regression(dem_voteshare))
    competitiveness_scores.append(competitiveness_score(natlDemVote[ELECTION_USED], dem_voteshare))
    iterations += 1
    if iterations%10 == 0: shrinking_variable *= shrink_ratio**(-4) #grace period half the time
    else: shrinking_variable *= shrink_ratio
    if iterations == TOTAL_STEPS: final_partition = partition


mean_expected = sum(expected_dems)/len(expected_dems)
deviation_from_expected = [round(abs(expected_dems[i] - mean_expected), 5) for i in range(len(expected_dems))]
sorted_expected_dev = sorted(deviation_from_expected)
init_expected_dems_idx = sorted_expected_dev.index(round(abs(expected_dems[0] - mean_expected), 5))
skew_percentile = 100*init_expected_dems_idx/len(expected_dems)
print(f'This plan is more skewed than {round(skew_percentile, 2)}% of the ensemble.')

total_polsby = [avg_polsby_raw[i]+min_polsby_raw[i] for i in range(len(expected_dems))]
init_polsby = total_polsby[0]
total_polsby.sort()
print(f'This plan is less compact than {round(100-100*total_polsby.index(init_polsby)/len(total_polsby))}% of the ensemble.')

weights = [20, 20, -2, 0.06, 20, 0.1]
composite_data = [avg_polsby_raw, min_polsby_raw, deviation_from_expected, geographic_scores, regression_scores, competitiveness_scores]
composite_scores = [sum([weights[metric]*composite_data[metric][i]
                        for metric in range(len(weights))]) for i in range(len(expected_dems))]


raw_data = [avg_polsby_raw, min_polsby_raw, expected_dems, geographic_scores, regression_scores, competitiveness_scores, composite_scores]

vote_data = DataFrame(unprocessed_vote_data)
# data = data[100:]
fig, ax = plt.subplots(2, (width := 4), figsize=(15, 8))

ax[0,0].axhline(0.5, color="#cccccc") # Draw 50% line

vote_data.boxplot(ax=ax[0,0], positions=range(len(vote_data.columns)))# Draw boxplot

# Draw initial plan's Democratic vote %s (.iloc[0] gives the first row)
#TODO change size of red dots
ax[0,0].plot(vote_data.iloc[0], "ro")

# Annotate
fig.suptitle("Comparing the 2011 plan to an ensemble")

ax[0,0].set_title("Vote by District")
ax[0,0].set_ylabel("Democratic vote % (Presidential 2016)")
ax[0,0].set_xlabel("Sorted districts")
ax[0,0].set_ylim(0, 1)
ax[0,0].set_yticks([0, 0.25, 0.5, 0.75, 1])

graphs = 7
n, bins, hist = [0 for i in range(graphs)], [0 for i in range(graphs)], [0 for i in range(graphs)]

# in order: avg polsby, min polsby, expected dem, geographic boundary, regression, competitiveness, composite
histogram_width_reciprocal = [200, 200, 10, 2, 100, 1, 3]

from math import floor, ceil
for graphplot in range(graphs):
    raw = raw_data[graphplot]
    reciprocal = histogram_width_reciprocal[graphplot]
    lowest_bin, highest_bin = floor(reciprocal*min(raw)), ceil(reciprocal*max(raw))
    trash_cans = [i/reciprocal for i in range(lowest_bin, highest_bin+1)]
    df_data = DataFrame(raw)
    x, y = (graphplot+1)//width, (graphplot+1)%width
    n[graphplot], bins[graphplot], hist[graphplot] = ax[x,y].hist(df_data, bins=trash_cans)
    start_bin = floor(reciprocal*raw[0]) -  lowest_bin
    hist[graphplot][start_bin].set_fc('r')
    ax[x,y].set_ylabel("Count")
    ax[x,y].set_xlabel("Score")

ax[0,1].set_title("Average Polsby-Popper")
ax[0,2].set_title("Smallest Polsby-Popper")
ax[0,3].set_title("Expected Democrat Seats")
ax[1,0].set_title("Geographic Score")
ax[1,1].set_title("Regression")
ax[1,2].set_title("Competitiveness Score")
ax[1,3].set_title("Composite Score")

fig.tight_layout(pad=4.0)

# display the districts

import geopandas
units = geopandas.read_file("PA/PA.shp")
units.to_crs({"init": "epsg:26986"}, inplace=True)
final_partition.plot(units, figsize=(10, 7), cmap="tab20")
plt.axis('off')

plt.show()