### Module for determining which color chips are above a certain threshold of centrality ###

import numpy as np
from os import path
from matplotlib import pyplot as plt
import pandas as pd


def convert_rowcol_to_index(chip):
    '''Converts a color chip representation from row, col to index.'''

    if chip[0] == 'A' or chip[0] == 'J' or chip[1:] =='0':
        return -1
    else:
        row_convert = {'B':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7}
        row = row_convert[chip[0]]
        col = int(chip[1:]) - 1
        return 40*row + col

def above_thresh(threshold):
    chips = np.where(props >= threshold, 1, 0)
    chips_grid = np.reshape(chips, (8,40))

    ax = None
    fig, ax = plt.subplots()

    ax.matshow(chips_grid, cmap="hot_r")

    plt.show()


foci_data = pd.read_table(path.abspath("all_lang_foci_results.txt"))
foci_data = np.array(foci_data)
for row in foci_data:
    for i in range(len(foci_data[0]) - len(row)):
        row.append("")
foci_data = np.array(foci_data)
focals = [x[3] for x in foci_data[1:] if x[4] != "" ]

achromatics = ['B0','C0','D0','E0','F0','G0','H0','I0']

counts = np.zeros(shape=320)

for focal in focals:
    if not('A' in focal or 'J' in focal or focal in achromatics):    #ignore any focal colors in the achromatic axis
        index = convert_rowcol_to_index(focal)
        counts[index] += 1

props = counts/max(counts)

counts_grid = np.reshape(counts, (8,40))
props_grid = np.reshape(props, (8,40))




















