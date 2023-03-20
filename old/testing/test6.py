import matplotlib.pyplot as plt
from gerrychain import (GeographicPartition, Partition, Graph, MarkovChain, proposals, updaters, constraints, accept, Election)
from gerrychain.proposals import recom
from functools import partial
import pandas

graph = Graph.from_json("PA/PA_VTDs.json")

elections = [
    Election("SEN10", {"Democratic": "SEN10D", "Republican": "SEN10R"}),
    Election("SEN12", {"Democratic": "USS12D", "Republican": "USS12R"}),
    Election("SEN16", {"Democratic": "T16SEND", "Republican": "T16SENR"}),
    Election("PRES12", {"Democratic": "PRES12D", "Republican": "PRES12R"}),
    Election("PRES16", {"Democratic": "T16PRESD", "Republican": "T16PRESR"})
]

POP_FIELD_NAME = "TOTPOP"
COUNTY_FIELD_NAME = "COUNTYFP10"
ELECTION_USED = "PRES16"
natlDemVote = {"PRES12": 51.964, "PRES16": 51.113}

# Population updater, for computing how close to equality the district
# populations are. "TOTPOP" is the population column from our shapefile.
my_updaters = {"population": updaters.Tally("TOTPOP", alias="population")}

# Election updaters, for computing election results using the vote totals
# from our shapefile.
election_updaters = {election.name: election for election in elections}
my_updaters.update(election_updaters)

initial_partition = GeographicPartition(graph, assignment="CD_2011", updaters=my_updaters)

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

compactness_bound = constraints.UpperBound(
    lambda p: len(p["cut_edges"]),
    2*len(initial_partition["cut_edges"])
)

pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, 0.02)

chain = MarkovChain(
    proposal=proposal,
    constraints=[
        pop_constraint,
        compactness_bound
    ],
    accept=accept.always_accept,
    initial_state=initial_partition,
    total_steps=20
)

# probability of winning; stdevs sourced from planscore at https://planscore.org/models/data/2022F/
# this adds about 1 seconds per 10000 runs
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

# how well does this partition preserve geographic boundaries (county lines)?
from gerrychain.updaters import county_splits # dict, maps to tuple (split, nodes, seen)
def geographic_boundary_score(partition, countyFieldName, populationFieldName, iterationNum): #examples: 'COUNTYFP10', 'CNTY_NAME'
    node_counties = (county_splits(iterationNum, countyFieldName)(partition))
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
        county_pop = sum([districtPop[dist] for dist in county_cds])
        county_pop2 = sum([districtPop[dist]**2 for dist in county_cds])
        random_guess_correct = county_pop2/(county_pop**2) # chance that a random guess of a resident's district is correct
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

unprocessed_vote_data = []
avg_polsby_raw, min_polsby_raw = [], []
expected_dems = []
geographic_scores = []
regression_scores = []
iteration = 0

for partition in chain.with_progress_bar():
    partition.name = str(iteration)
    dem_voteshare = sorted(partition[ELECTION_USED].percents("Democratic"))
    unprocessed_vote_data.append(dem_voteshare)
    avg_polsby_raw.append(overall_polsby_popper(partition))
    min_polsby_raw.append(min_polsby_popper(partition))
    expected_dems.append(expectedDem(natlDemVote[ELECTION_USED], dem_voteshare))
    geographic_scores.append(geographic_boundary_score(partition, COUNTY_FIELD_NAME, POP_FIELD_NAME, str(iteration)))
    regression_scores.append(regression(dem_voteshare))
    iteration += 1

vote_data = pandas.DataFrame(unprocessed_vote_data)
# data = data[100:]
fig, ax = plt.subplots(2, (width := 3), figsize=(8, 6))

ax[0,0].axhline(0.5, color="#cccccc") # Draw 50% line

vote_data.boxplot(ax=ax[0,0], positions=range(len(vote_data.columns)))# Draw boxplot

# Draw initial plan's Democratic vote %s (.iloc[0] gives the first row)
#TODO change size of red dots
ax[0,0].plot(vote_data.iloc[0], "ro")

# Annotate
ax[0,0].set_title("Comparing the 2011 plan to an ensemble")
ax[0,0].set_ylabel("Democratic vote % (Presidential 2016)")
ax[0,0].set_xlabel("Sorted districts")
ax[0,0].set_ylim(0, 1)
ax[0,0].set_yticks([0, 0.25, 0.5, 0.75, 1])

graphs = 5
n, bins, hist = [0 for i in range(graphs)], [0 for i in range(graphs)], [0 for i in range(graphs)]

# in order: avg polsby, min polsby, expected dem, geographic boundary, regression
raw_data = [avg_polsby_raw, min_polsby_raw, expected_dems, geographic_scores, regression_scores]
histogram_width_reciprocal = [50, 50, 2, 1, 100]

for graphplot in range(graphs):
    raw = raw_data[graphplot]
    reciprocal = histogram_width_reciprocal[graphplot]
    lowest_bin, highest_bin = round(reciprocal*min(raw)), round(reciprocal*max(raw))+1
    trash_cans = [i/reciprocal for i in range(lowest_bin, highest_bin+1)]
    df_data = pandas.DataFrame(raw)
    x, y = (graphplot+1)//width, (graphplot+1)%width
    n[graphplot], bins[graphplot], hist[graphplot] = ax[x,y].hist(df_data, bins=trash_cans)
    start_bin = round(reciprocal*raw[0]) -  lowest_bin
    hist[graphplot][start_bin].set_fc('r')

# # avg polsby
# lowest_bin_apd, highest_bin_apd = round(50*min(avg_polsby_raw)), round(50*max(avg_polsby_raw))+1
# trash_cans_apd = [i/50 for i in range(lowest_bin_apd, highest_bin_apd+1)]
# avg_polsby_data = pandas.DataFrame(avg_polsby_raw)
# n_apd, bins_apd, hist_apd = ax[0,1].hist(avg_polsby_data, bins=trash_cans_apd)
# start_bin_apd = round(50*avg_polsby_raw[0]) - lowest_bin_apd
# hist_apd[start_bin_apd].set_fc('r')

# # min polsby
# lowest_bin_mpd, highest_bin_mpd = round(50*min(min_polsby_raw)), round(50*max(min_polsby_raw))+1
# trash_cans_mpd = [i/50 for i in range(lowest_bin_mpd, highest_bin_mpd+1)]
# min_polsby_data = pandas.DataFrame(min_polsby_raw)
# n_mpd, bins_mpd, hist_mpd = ax[0,2].hist(min_polsby_data, bins=trash_cans_mpd)
# start_bin_mpd = round(50*min_polsby_raw[0]) - lowest_bin_mpd
# hist_mpd[start_bin_mpd].set_fc('r')

# # expected dem
# lowest_bin_exd, highest_bin_exd = round(2*min(expected_dems)), round(2*max(expected_dems))+1
# trash_cans_exd = [i/2 for i in range(lowest_bin_exd, highest_bin_exd+1)]
# expected_dems_df = pandas.DataFrame(expected_dems)
# n_exd, bins_exd, hist_exd = ax[1,0].hist(expected_dems_df, bins = trash_cans_exd)
# start_bin_exd = round(2*expected_dems[0]) - lowest_bin_exd
# hist_exd[start_bin_exd].set_fc('r')

# geographic boundary score


plt.show()