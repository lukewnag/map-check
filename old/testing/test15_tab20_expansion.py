colors40 = ["#c21f5b","#fbc02c","#ff5722","#bf360c","#fdf9c3","#c5cae9","#fff177","#f8bbd0","#e1bee7","#3f51b5","#aed581",
          "#253137","#ba68c8","#ffccbc","#9c27b0","#90a4ae","#e92663","#0488d1","#7986cb","#e64a18","#03a9f4","#f06292",
          "#f67f17","#19237e","#03579b","#cfd8dc","#ddedc8","#689f38","#607d8b","#b3e5fc","#303f9f","#33691d","#455a64",
          "#88144f","#8bc34a","#ff8a65","#4a198c","#7b21a2","#4fc3f7","#ffec3a"]

from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colorbar
from matplotlib import pyplot as plt

bar = LinearSegmentedColormap.from_list("custom", colors40)


figMap, axMap = plt.subplots(figsize=(10, 3))
min_colors = colors40
bar = LinearSegmentedColormap.from_list("custom", min_colors)
colorbar.ColorbarBase(ax=axMap, cmap=bar, orientation = 'horizontal')

plt.show()