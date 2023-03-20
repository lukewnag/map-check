# probability of winning; stdevs sourced from planscore at https://planscore.org/models/data/2022F/
# this adds about 1 seconds per 10000 runs
from statistics import NormalDist
SAMPLES = 1000 # number of national elections we want to simulate for overall environment
natl_stdev = 2 # how much does the nation swing from election to election?
offsets = [natl_stdev*NormalDist().inv_cdf(i/(SAMPLES+1)) for i in range(1, SAMPLES+1)]
def probDem(natlDemVote, districtDemVote): # natlDemVote is the dem 2 party vote share
    election_stdev = 3 # percent swing in dem vote - district level
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
def geographic_boundary_score(partition, countyFieldName, populationFieldName): #examples: 'COUNTYFP10', 'CNTY_NAME'
    node_counties = (county_splits('partition_name', countyFieldName)(partition))
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