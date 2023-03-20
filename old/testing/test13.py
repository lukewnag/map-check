from matplotlib import pyplot as plt
from matplotlib import colorbar
from matplotlib.colors import LinearSegmentedColormap

colors = [(0, (0,0,0)), (0.5, (0.9,0.9,0.9)), (0.6, (0.5,0.7,0)), (1, (0.25, 0.35, 0))]
bar = LinearSegmentedColormap.from_list("custom", colors)

figMap, axMap = plt.subplots(2,3, figsize=(10, 3), height_ratios=[1,3])
figMap.tight_layout(pad=1)
colorbars = [[(0, (0,0,0)), (0.5, (0.9,0.9,0.9)), (0.6, (0.5,0.7,0)), (1, (0.25, 0.35, 0))],
                [(0, (0,0,0)), (0.5, (0,1,1)), (1, (0.5,1,1))],
                [(0, (0.2,0,0)), (0.4, (0.8,0,0)), (0.5, (0.8,0.8,0.8)), (0.6, (0,0,0.8)), (1, (0,0,0.2))]]
colorbar_desc = ['Minority Percent (Non-Opportunity)', 'Minority Percent (Opportunity)', 'Democrat Vote Share']

for col in range(3):
    min_colors = colorbars[col]
    bar = LinearSegmentedColormap.from_list("custom", min_colors)
    axMap[0, col].set_title(colorbar_desc[col])
    colorbar.ColorbarBase(ax=axMap[0,col], cmap=bar, orientation = 'horizontal')

plt.show()