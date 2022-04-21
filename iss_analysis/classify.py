import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
import pciSeq
from .io import load_data_tasic_2018
from itertools import cycle


def classify_cells(masks_file, spots_data, ref_data):
    masks = np.load(masks_file)

    with np.load(spots_data, allow_pickle=True) as data:
        rolony_locations = data['rolony_locations'].tolist()
        rolony_genes = data['gene_names'].tolist()
    for rolony_location, gene_name in zip(rolony_locations, rolony_genes):
        rolony_location['Gene'] = gene_name
    spots = pd.concat(rolony_locations, ignore_index=True)
    rolony_genes.remove('Ccn2')
    rolony_genes.remove('Tafa1')
    rolony_genes.remove('Tafa2')
    spots = spots[(spots['Gene'] != 'Ccn2') & (spots['Gene'] != 'Tafa1') & (spots['Gene'] != 'Tafa2')]

    exons_df, genes = load_data_tasic_2018(ref_data)
    sc_data = exons_df.set_index('cluster').filter(regex='\d').set_axis(genes, axis=1, inplace=False)

    opts = {}
    opts['Inefficiency'] = 0.001
    opts['MisreadDensity'] = 0.0000001
    opts['SpotReg'] = 0.01
    cellData, geneData = pciSeq.fit(spots, coo_matrix(masks), sc_data[rolony_genes].T, opts=opts)
    cellData['BestClass'] = cellData.apply(lambda r: r['ClassName'][np.argmax(r['Prob'])], axis=1)
    cellData['BestProb'] = cellData.apply(lambda r: np.max(r['Prob']), axis=1)
    return cellData, geneData


def plot_cell_types(cellData):
    colors = {
        'L2/3': 'deepskyblue',
        'L4': 'dodgerblue',
        'L5 IT': 'blue',
        'L5 NP': 'magenta',
        'L5 PT': 'blueviolet',
        'L6 CT':  'forestgreen',
        'L6 IT': 'violet',
        'L6b': 'black',
        'Pvalb': 'darkorange',
        'Sst':  'orangered',
        'Sncg': 'deeppink',
        'Serpinf1': 'limegreen',
        'Lamp5': 'tomato',
        'Vip': 'crimson'
    }
    markers = cycle('ov^<>spPXD*')
    plt.figure(figsize=(15,15))
    cellData = cellData[cellData['BestClass'] != 'Zero']
    ax = plt.subplot(1,1,1)
    clusters = np.sort(cellData['BestClass'].unique())
    for i, cluster in enumerate(clusters):
        color = 'gray'
        for type in colors:
            if cluster.find(type) != -1:
                color = colors[type]
        plt.plot(
            cellData[cellData['BestClass'] == cluster]['X'],
            cellData[cellData['BestClass'] == cluster]['Y'],
            next(markers),
            c=color,
            markersize=10
        )

    plt.legend(clusters, loc='right', ncol=2, bbox_to_anchor=(0.,0.,2.5,1.))
    ax.set_aspect('equal', 'box')
    ax.invert_yaxis()