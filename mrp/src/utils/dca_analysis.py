# ---- install -----
# install dcurves to perform DCA (first install package via pip)
# pip install dcurves
from dcurves import dca, plot_graphs
import matplotlib
# matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
# install other packages used in this tutorial
# pip install pandas numpy statsmodels lifelines
import pandas as pd
import numpy as np
import statsmodels.api as sm
import lifelines
import numpy as np
from sklearn.metrics import roc_curve, auc

# palette = plt.get_cmap('Paired')
palette = plt.get_cmap('tab10')


MRP_predictions_average = pd.read_csv('/dcia_combined_allmodels.csv')
# #statistics
# post_predi = np.array(MRP_predictions_average['MRP_Post_'])
# pre_predi = np.array(MRP_predictions_average['MRP_pre_'])
# dcis_pcr = np.array(MRP_predictions_average['pcr'])
# pure_pcr = np.array(MRP_predictions_average['purepcr'])
# fpr, tpr, thresholds = roc_curve(dcis_pcr, pre_predi)
# for (sensi,speci) in zip(tpr,fpr):
#     if sensi==1:  #sensitity =1, FN=0, without missing pcr
#         print("speci for pre, without missing pcr", 1-speci)
# roc_auc = auc(fpr, tpr)
#
# fpr2, tpr2, thresholds2 = roc_curve(pure_pcr, post_predi)
# for (speci,sensi) in zip(fpr2,tpr2):
#     if speci==0: #specificity =1, FP=0, without missing non-pcr
#         print("sensity for post, without missing non-pcr", sensi)
# roc_auc2 = auc(fpr2, tpr2)

df_cancer_dx = MRP_predictions_average

# ---- dca_intervention -----
modelnames = [['Reader_1_Pre_NAT', 'Reader_2_Pre_NAT', 'Reader_3_Pre_NAT', 'Reader_4_Pre_NAT', 'Reader_5_Pre_NAT', 'Reader_6_Pre_NAT', 'MRP_Pre_NAT'],
              ['Reader_1_Pre_NAT','Reader_2_Pre_NAT', 'Reader_3_Pre_NAT', 'Reader_4_Pre_NAT', 'Reader_5_Pre_NAT', 'Reader_6_Pre_NAT', 'rhpc', 'MG_rhpc_Pre_NAT','MR_rhpc_Pre_NAT','MRP_Pre_NAT'],
              ['Reader_1_Mid_NAT', 'Reader_2_Mid_NAT', 'Reader_3_Mid_NAT', 'Reader_4_Mid_NAT', 'Reader_5_Mid_NAT', 'Reader_6_Mid_NAT', 'MRP_Mid_NAT'],
              ['Reader_1_Mid_NAT', 'Reader_2_Mid_NAT', 'Reader_3_Mid_NAT', 'Reader_4_Mid_NAT', 'Reader_5_Mid_NAT', 'Reader_6_Mid_NAT', 'rhpc', 'MG_rhpc_Pre_NAT', 'MR_rhpc_Mid_NAT', 'MRP_Mid_NAT'],
              ['Reader_1_Post_NAT', 'Reader_2_Post_NAT', 'Reader_3_Post_NAT', 'Reader_4_Post_NAT', 'Reader_5_Post_NAT', 'Reader_6_Post_NAT', 'MRP_Post_NAT'],
              ['Reader_1_Post_NAT','Reader_2_Post_NAT', 'Reader_3_Post_NAT', 'Reader_4_Post_NAT', 'Reader_5_Post_NAT', 'Reader_6_Post_NAT','rhpc','MG_rhpc_Pre_NAT','MR_rhpc_Post_NAT','MRP_Post_NAT']]
outputdirs = ['/plot/dca/mrp_reader/netintervention',
             '/plot/dca/ais_reader/netintervention',
             '/plot/dca/mrp_reader/netintervention',
             '/plot/dca/ais_reader/netintervention',
             '/plot/dca/mrp_reader/netintervention',
             '/plot/dca/ais_reader/netintervention']
colornames = [[palette(0), palette(1), palette(2), palette(4), palette(5), palette(6), palette(3), '#F9DA3B', 'black'],
               [palette(0), palette(1), palette(2), palette(4), palette(5), palette(6), palette(7), palette(8), palette(9), palette(3), '#F9DA3B', 'black'],
               [palette(0), palette(1), palette(2), palette(4), palette(5), palette(6), palette(3), '#F9DA3B', 'black'],
               [palette(0), palette(1), palette(2), palette(4), palette(5), palette(6), palette(7), palette(8), palette(9), palette(3), '#F9DA3B', 'black'],
               [palette(0), palette(1), palette(2), palette(4), palette(5), palette(6), palette(3), '#F9DA3B', 'black'],
               [palette(0), palette(1), palette(2), palette(4), palette(5), palette(6), palette(7), palette(8), palette(9), palette(3), '#F9DA3B', 'black'],
              ]
thresholds =[np.arange(0.05, 0.3, 0.01), np.arange(0.05, 0.3, 0.01), np.arange(0.05, 0.3, 0.01),
             np.arange(0.05, 0.3, 0.01), np.arange(0.05, 0.3, 0.01), np.arange(0.05, 0.3, 0.01)]
ylimits = [[0, 0.6],[0, 0.6],
           [0, 0.6],[0, 0.6],
           [0, 0.6],[0, 0.6]]
for index in range(6):
    dca_intervention_df = \
        dca(
            data=df_cancer_dx,
            outcome='pcr',

            modelnames=modelnames[index],
            # modelnames=['Reader_1_Mid_NAT', 'Reader_2_Mid_NAT', 'rhpc', 'MG_rhpc_Pre_NAT', 'MR_rhpc_Mid_NAT',
            #             'MRP_Mid_NAT'],
            # modelnames=['Reader_1_Pre_NAT','Reader_2_Pre_NAT','rhpc','MG_rhpc_Pre_NAT','MR_rhpc_Pre_NAT','MRP_Pre_NAT'],
            thresholds=thresholds[index],
            ##########post
            # modelnames=['Reader_1_Post_NAT','Reader_2_Post_NAT','rhpc','MG_rhpc_Pre_NAT','MR_rhpc_Post_NAT','MRP_Post_NAT'],
            # modelnames=['Reader_1_Post_NAT','Reader_2_Post_NAT', 'MRP_Post_NAT'],
            # thresholds=np.arange(0.05, 0.36, 0.01),
            # data= MRP_predictions_average_nonpcr,
            # outcome='non-pcr',
            # thresholds=np.arange(0.2, 0.3, 0.01),
        )

    plot_graphs(
        plot_df=dca_intervention_df,
        outputdir=outputdirs[index] + modelnames[index][0].split('1_')[1] + '.pdf',
        graph_type='net_intervention_avoided',
        y_limits = ylimits[index],
        ######post
        # y_limits = [0, 0.05],
        # y_limits = [-1, 0.2],
        # color_names = [palette(1), palette(3), palette(11), palette(9), palette(7), palette(5), 'black', 'yellow']
        color_names = colornames[index]
    )
