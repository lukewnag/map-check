from gerrychain import Graph, Partition, Election
from gerrychain.updaters import Tally, cut_edges

graph = Graph.from_json("PA/PA_aspose.geojson")

election = Election("SEN12", {"Dem": "USS12D", "Rep": "USS12R"})

initial_partition = Partition(
    graph,
    assignment="CD_2011",
    updaters={
        "cut_edges": cut_edges,
        "population": Tally("TOTPOP", alias="population"),
        "SEN12": election
    }
)

from gerrychain import MarkovChain
from gerrychain.constraints import single_flip_contiguous
from gerrychain.proposals import propose_random_flip
from gerrychain.accept import always_accept

steps = 20000

chain = MarkovChain(
    proposal=propose_random_flip,
    constraints=[single_flip_contiguous],
    accept=always_accept,
    initial_state=initial_partition,
    total_steps=steps
)

init, final = [], []

samples = 5
chainsThru = 0
for partition in chain.with_progress_bar():
    if chainsThru == 0: init = sorted(partition["SEN12"].percents("Dem"))
    chainsThru += 1
    if chainsThru == steps: final = sorted(partition["SEN12"].percents("Dem"))
    # if chainsThru%(skip := steps//samples) == skip-1: print(sorted(partition["SEN12"].percents("Dem")))

diff = [round(100 * (final[i] - init[i]), 4) for i in range(len(init))] # in percents
print(diff)