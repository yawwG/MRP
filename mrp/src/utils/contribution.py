import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

output_folder = '/utils/'

her2='agnost'
feats='chemo'
rcut = 0.8
df_labels_agnost_summ = pd.read_csv('/contribution_MRP.csv')

# Colorbar
cbar_f, cbar_ax = plt.subplots(1,1,figsize=(1,4))
cbar_ax.set_xticks([])

# Main figure
fig, ax = plt.subplots(1,3,gridspec_kw={'width_ratios': [1, 0.5, 0.5]},figsize=(13,12))
fig.subplots_adjust(hspace=0.00)

sns.heatmap(df_labels_agnost_summ[['mean10']], linewidths=5, cmap='Blues', ax=ax[0], cbar=True, vmax=2, cbar_ax=cbar_ax)
ax[0].set_xticklabels(['Importance z-score'], rotation=90)
ax[0].set_yticklabels(df_labels_agnost_summ['fetures'], rotation=0)

sns.heatmap(df_labels_agnost_summ[['stage']], linewidths=5, cmap='Set2', ax=ax[1], cbar=False)
ax[1].set_xticklabels(['Biological process'], rotation=90)
ax[1].set_yticks([])

sns.heatmap(df_labels_agnost_summ[['class']], linewidths=5, cmap='tab10_r', ax=ax[2], cbar=False)
ax[2].set_xticklabels(['Feature class'], rotation=90)
ax[2].set_yticks([])

cbar_f.savefig(output_folder+'/summary_feat_import_unsigned_averaged_colorbar.pdf', bbox_inches='tight', transparent=True)
fig.savefig(output_folder+'/summary_feat_import_unsigned_averaged.png', bbox_inches='tight', transparent=True, dpi=300)
plt.show()