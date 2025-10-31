import pandas as pd

seeds_df = pd.read_excel('seeds-less-rows.xlsx')
varieties = list(seeds_df.pop('grain_variety'))
samples = seeds_df.values

from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
mergings = linkage(samples, method='complete')
dendrogram(mergings,labels=varieties,leaf_rotation=90,leaf_font_size=6)
plt.show()

from scipy.cluster.hierarchy import fcluster
labels = fcluster(mergings, 6, criterion='distance')
df = pd.DataFrame({'labels': labels, 'varieties': varieties})
ct = pd.crosstab(df['labels'], df['varieties'])
ct
