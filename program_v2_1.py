# VERSION 2.1: display adds most competitive map, uses two windows to display maps to decrease crowding

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
        score += 2*probDemLookup(50, losing_pct) #score can max out at 100
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

def minority_opportunity(partition, POP_FIELD_NAME, WHITE_POP, dem_vote, gop_vote):
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
                dem += partition.graph.nodes[int(node)][dem_vote]
                gop += partition.graph.nodes[int(node)][gop_vote]
            if dem >= 0.53*(dem+gop) and minority_pop/tot_pop >= 0.625*dem/(dem+gop):
                opportunity_districts.add(district)
    return opportunity_districts

from gerrychain.metrics import partisan
from math import floor
from matplotlib.colors import LinearSegmentedColormap
import geopandas
from matplotlib import colorbar

# MARKOV CHAIN ANALYSIS

def analysis(STATE, TOTAL_STEPS, ELECTION_USED):
    
    YEAR = "2011"

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

    initial_partition = GeographicPartition(graph, assignment=ASSIGNMENT, updaters=my_updaters)
    # print(overall_polsby_popper(initial_partition))

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

    # polsby_goal = 0.23 #this is a fairly low value
    # starting_polsby = overall_polsby_popper(initial_partition)
    # def compactness_bound(partition):
    #     if starting_polsby >= polsby_goal: lower_bound = polsby_goal
    #     else: lower_bound = polsby_goal - (polsby_goal - starting_polsby * 0.9)* shrinking_variable
    #     return overall_polsby_popper(partition) > lower_bound
    
    starting_cut_edges = len(initial_partition["cut_edges"])
    def compactness_bound(partition):
        return len(partition['cut_edges']) < 1.15*starting_cut_edges

    pop_constraint = constraints.within_percent_of_ideal_population(initial_partition, 0.02)

    chain = MarkovChain(
        proposal=proposal,
        constraints=[
            pop_constraint,
            compactness_bound
        ],
        accept=accept.always_accept,
        initial_state=initial_partition,
        total_steps=TOTAL_STEPS
    )
# az: 42.740 vote, nat lean = -2.572 --> 45.312
# co: 55.509 vote, nat lean = 3.108 --> 52.401
# mi: 54.892 vote, nat lean = -1.056 --> 55.948
# or: 53.412, nat lean = 5.527 --> 47.885
    gov_leans = {'AZ': 45.312, 'CO': 52.401, 'MI': 55.948, 'OR': 47.885}
    if "PRES" in ELECTION_USED or "HOUSE" in ELECTION_USED: lean = natlDemVote[ELECTION_USED]
    else: lean = gov_leans[STATE]

    unprocessed_vote_data = []
    minority_raw = []
    avg_polsby_raw, min_polsby_raw = [], []
    expected_dems = []
    geographic_scores = []
    regression_scores = []
    competitiveness_scores = []
    efficiency_gaps, gini_scores = [], []

    regression_partition, minority_partition, competitive_partition = "", "", ""
    
    for partition in chain.with_progress_bar():
        dem_voteshare = sorted(partition[ELECTION_USED].percents("Democratic"))
        unprocessed_vote_data.append(dem_voteshare)
        minority_raw.append(len(minority_opportunity(partition, POP_FIELD_NAME, WHITE_POP, dem_vote, gop_vote)))
        avg_polsby_raw.append(overall_polsby_popper(partition))
        min_polsby_raw.append(min_polsby_popper(partition))
        expected_dems.append(expectedDem(lean, dem_voteshare))
        geographic_scores.append(geographic_boundary_score(partition, COUNTY_FIELD_NAME, POP_FIELD_NAME))
        regression_scores.append(regression(dem_voteshare))
        competitiveness_scores.append(competitiveness_score(lean, dem_voteshare))
        efficiency_gaps.append(100*partisan.efficiency_gap(partition[ELECTION_USED]))
        gini_scores.append(partisan.partisan_gini(partition[ELECTION_USED]))
        iterations += 1
        if iterations%10 == 0: shrinking_variable *= shrink_ratio**(-4) #grace period half the time
        else: shrinking_variable *= shrink_ratio
        # if iterations == TOTAL_STEPS: final_partition = partition
        if regression_scores[-1] == max(regression_scores): regression_partition = partition
        if minority_raw[-1] == max(minority_raw): minority_partition = partition
        if competitiveness_scores[-1] == max(competitiveness_scores): competitive_partition = partition


    mean_expected = sum(expected_dems)/len(expected_dems)
    deviation_from_expected = [round(abs(expected_dems[i] - mean_expected), 5) for i in range(len(expected_dems))]
    sorted_expected_dev = sorted(deviation_from_expected)
    init_expected_dems_idx = sorted_expected_dev.index(round(abs(expected_dems[0] - mean_expected), 5))
    skew_percentile = init_expected_dems_idx/len(expected_dems)
    print(f'This plan is more skewed than {round(100*skew_percentile, 2)}% of the ensemble.')

    total_polsby = [avg_polsby_raw[i]+min_polsby_raw[i] for i in range(len(expected_dems))]
    init_polsby = total_polsby[0]
    total_polsby.sort()
    print(f'This plan is less compact than {round(100-100*total_polsby.index(init_polsby)/len(total_polsby))}% of the ensemble.')

    efficiency_deviation = [abs(efficiency_gaps[i]) for i in range(len(efficiency_gaps))]

    weights = [20/len(partition.parts), 4, 10, -20/len(partition.parts), 0.05, (len(partition.parts)*50)**0.5, 0.15, -0.2, -25] #TODO
    composite_data = [minority_raw, avg_polsby_raw, min_polsby_raw, deviation_from_expected, geographic_scores,
                      regression_scores, competitiveness_scores, efficiency_deviation, gini_scores]
    composite_scores = [sum([weights[metric]*composite_data[metric][i]
                            for metric in range(len(weights))]) for i in range(len(expected_dems))]


    raw_data = [minority_raw, avg_polsby_raw, min_polsby_raw, expected_dems, geographic_scores, regression_scores,
                competitiveness_scores, efficiency_gaps, gini_scores, composite_scores]

    vote_data = DataFrame(unprocessed_vote_data)
    # data = data[100:]
    fig, ax = plt.subplots(2, (width := 5), figsize=(14, 6))
    figBP, axBoxplot = plt.subplots(figsize = (8, 6))

    axBoxplot.axhline(lean/100, color="r") # Draw 50% line

    vote_data.boxplot(ax=axBoxplot, positions=range(len(vote_data.columns)))# Draw boxplot

    # Draw initial plan's Democratic vote %s (.iloc[0] gives the first row)
    axBoxplot.plot(vote_data.iloc[0], "ro")

    # Annotate
    fig.suptitle("Comparing the "+YEAR+" "+STATE+" Congressional plan to an ensemble of "+str(TOTAL_STEPS))

    axBoxplot.set_title(f"Vote in the {ELECTION_USED} Election by District, {STATE}")
    axBoxplot.set_ylabel("Democratic vote % (Presidential 2016)")
    axBoxplot.set_xlabel("Sorted districts")
    axBoxplot.set_ylim(0, 1)
    axBoxplot.set_yticks([0, 0.25, 0.5, 0.75, 1])

    graphs = 10
    n, bins, hist = [0 for i in range(graphs)], [0 for i in range(graphs)], [0 for i in range(graphs)]

    # in order: avg polsby, min polsby, expected dem, geographic boundary, regression, competitiveness, composite
    # histogram_width_reciprocal = [200, 200, 10, 2, 100, 1, 3]

    # bar graph of minority opportunity districts
    minority_values = [*range(min(minority_raw), max(minority_raw)+1)]
    count = [minority_raw.count(num) for num in minority_values]
    colors = ['red' if i == minority_raw[0] else '#1F77B4' for i in minority_values]
    ax[0,0].bar(x=minority_values, height=count, width=0.8, color=colors)

    
    for graphplot in range(1, graphs):
        raw = raw_data[graphplot]
        # reciprocal = histogram_width_reciprocal[graphplot]
        bin_width = (max(raw)-min(raw))/19.99 # so maximum bar can still be represented despite arithmetic error
        # lowest_bin, highest_bin = floor(reciprocal*min(raw)), ceil(reciprocal*max(raw))
        trash_cans = [min(raw)+i*bin_width for i in range(21)]
        df_data = DataFrame(raw)
        x, y = (graphplot)//width, (graphplot)%width
        n[graphplot], bins[graphplot], hist[graphplot] = ax[x,y].hist(df_data, bins=trash_cans)
        start_bin = floor((raw[0] - min(raw))/bin_width)
        hist[graphplot][start_bin].set_fc('r')
        ax[x,y].set_ylabel("Count")
        ax[x,y].set_xlabel("Score")
        if n[graphplot][start_bin] < 0.02*max(n[graphplot]):
            ax[x,y].axvline((start_bin+0.5)*bin_width+min(raw), color="r", linestyle='dashed', 
                            ymax = 1, linewidth = 1.5, label = "Enacted plan") # Draw line

    ax[0,0].set_title("Minority Representation")
    ax[0,1].set_title("Average Polsby-Popper")
    ax[0,2].set_title("Smallest Polsby-Popper")
    ax[0,3].set_title("Expected Democrat Seats")
    ax[0,4].set_title("Geographic Score")
    ax[1,0].set_title("Regression")
    ax[1,1].set_title("Competitiveness Score")
    ax[1,2].set_title("Efficiency Gap")
    ax[1,3].set_title("Partisan Gini Score")
    ax[1,4].set_title("Composite Score")

    fig.tight_layout(pad=3.0)

    # display the districts
    
    # heatmap_colors = [(0, (0,0,0)), (0.5, (0.9,0.9,0.9)), (0.6, (0.5,0.7,0)), (1, (0.25, 0.35, 0))]
    # different color if it's a minority opportunity district
    def minority_coloring(partition):
        minority_proportions = {}
        for district in partition.parts:
            tot_pop, minority_pop = 0, 0
            for node in partition.parts[district]:
                tot_pop += partition.graph.nodes[int(node)][POP_FIELD_NAME]
                minority_pop += partition.graph.nodes[int(node)][POP_FIELD_NAME] - partition.graph.nodes[int(node)][WHITE_POP]
            minority_proportions[district] = minority_pop/tot_pop
        
        district_colors = [0 for i in range(len(partition.parts))]
        districts = [int(district) for district in partition.parts]
        minority_districts = minority_opportunity(partition, POP_FIELD_NAME, WHITE_POP, dem_vote, gop_vote)
        for district in partition.parts:
            if 0 in districts: district_num = int(district)
            else: district_num = int(district)-1
            minority_percent = minority_proportions[district]
            if district in minority_districts: # it's minority controlled
                scaler = min(1, minority_percent*2)
                dark = min(0, minority_percent-0.5)
                district_colors[district_num] = (district_num/(len(partition.parts)-1), (dark, scaler, scaler))
            elif minority_percent < 0.5:
                gray = 1.8*minority_percent
                district_colors[district_num] = (district_num/(len(partition.parts)-1), (gray, gray, gray))
            elif minority_percent < 0.6:
                minority_percent -= 0.5
                red = 0.9 - 4*minority_percent
                green = 0.9 - 2*minority_percent
                blue = 0.9 - 9*minority_percent
                district_colors[district_num] = (district_num/(len(partition.parts)-1), (red, green, blue))
            else: # kept here just in case I need to remove the first condition later
                minority_percent -= 0.6
                red = 0.5 - 0.625*minority_percent
                green = 0.7 - 0.875*minority_percent
                blue = 0
                district_colors[district_num] = (district_num/(len(partition.parts)-1), (red, green, blue))

        return LinearSegmentedColormap.from_list("custom", district_colors)

    # heatmap_colors = [(0, (0.8,0,0)), (0.5, (0.8,0.8,0.8)), (1, (0,0,0.8))]
    def partisan_coloring(partition):
        dem_proportions = {}
        for district in partition.parts:
            num_dem_votes, num_gop_votes = 0, 0
            for node in partition.parts[district]:
                num_dem_votes += partition.graph.nodes[int(node)][dem_vote]
                num_gop_votes += partition.graph.nodes[int(node)][gop_vote]
            dem_proportions[district] = num_dem_votes/(num_dem_votes+num_gop_votes)+(50-lean)/100 # adjusts for nat'l environment
        
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

    
    units = geopandas.read_file(''.join([STATE, YEAR, '/', STATE, '.shp']))
    units.to_crs({"init": "epsg:"+projection_code[STATE]}, inplace=True)

    figMap, axMap = plt.subplots(3, 3, figsize = (14, 7), height_ratios = [10,10,1])
    figMap.subplots_adjust(top=0.8, bottom=0.1)
    figMap.suptitle(f"{STATE} Enacted Plan of {YEAR} (Top) and an Ensemble-Drawn Plan Maximizing Minority Representation (Bottom)\n ")
    
    figMap2, axMap2 = plt.subplots(3, 3, figsize = (14, 7), height_ratios = [10,10,1])
    figMap2.subplots_adjust(top=0.8, bottom=0.1)
    figMap2.suptitle(f"{STATE} Ensemble-Drawn Plans for {YEAR} Maximizing Regression (Top) and Competitiveness (Bottom)\n ")
    
    row = 0
    for partition in [initial_partition, minority_partition]:
        partition.plot(units, ax=axMap[row,0], cmap="tab20")
        partition.plot(units, ax=axMap[row,1], cmap=minority_coloring(partition))
        partition.plot(units, ax=axMap[row,2], cmap=partisan_coloring(partition))
        row += 1
    row = 0
    for partition in [regression_partition, competitive_partition]:
        partition.plot(units, ax=axMap2[row,0], cmap="tab20")
        partition.plot(units, ax=axMap2[row,1], cmap=minority_coloring(partition))
        partition.plot(units, ax=axMap2[row,2], cmap=partisan_coloring(partition))
        row += 1

    figMap.tight_layout(pad=0)
    figMap2.tight_layout(pad=0)
    for i in range(2):
        for j in range(3):
            axMap[i,j].axis('off')
            axMap2[i,j].axis('off')
    
    colorbars = [[(0, (0,0,0)), (0.5, (0.9,0.9,0.9)), (0.6, (0.5,0.7,0)), (1, (0.25, 0.35, 0))],
                 [(0, (0,0,0)), (0.5, (0,1,1)), (1, (0.5,1,1))],
                 [(0, (0.2,0,0)), (0.4, (0.8,0,0)), (0.5, (0.8,0.8,0.8)), (0.6, (0,0,0.8)), (1, (0,0,0.2))]]
    colorbar_desc = ['Minority Percent (Non-Opportunity)', 'Minority Percent (Opportunity)', 'Democrat Vote Share']
    for col in range(3):
        min_colors = colorbars[col]
        bar = LinearSegmentedColormap.from_list("custom", min_colors)
        axMap[2, col].set_title(colorbar_desc[col])
        colorbar.ColorbarBase(ax=axMap[2,col], cmap=bar, orientation = 'horizontal')
        axMap2[2, col].set_title(colorbar_desc[col])
        colorbar.ColorbarBase(ax=axMap2[2,col], cmap=bar, orientation = 'horizontal')

# # OVERNIGHT RUNS
# analysis('AZ', 15000, 'GOV18') # cleared; error with shapefile: very many overlaps among polygons
# analysis('CO', 6000, 'GOV18') # cleared
# analysis('GA', 10000, 'PRES16') # cleared
# analysis('MI', 8000, 'PRES16') # cleared - remade the json
# analysis('MN', 6000, 'PRES16') # cleared
# analysis('NC', 12000, 'PRES16') # cleared; error with shapefile
# analysis('OH', 5000, 'PRES16') # cleared
# # analysis('OR', 30, 'GOV18') # 2nd district has polsby popper of 5.511766159282041..... need to fix
# analysis('PA', 5000, 'PRES16') # cleared
# analysis('TX', 8000, 'PRES16') # cleared; error with shapefile
# analysis('WI', 3000, 'PRES16') # cleared

# TEST
analysis('NC', 10000, 'PRES16')

# analysis('GA', 10000, 'PRES16') # cleared
# analysis('MI', 8000, 'PRES16') # cleared - remade the json
# analysis('WI', 5000, 'PRES16') # cleared

plt.show()