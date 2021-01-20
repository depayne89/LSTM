import numpy as np
import torch
import scipy.stats as stats
import matplotlib.pyplot as plt

import metrics as met


batch_size = 16
learning_rate = .001
trans_text = '_spec'

min_c1 = 16  # 15
min_c2 = 32  # 64
min_c3 = 64
min_fc1 = 32  # 128
min_fc2 = 16

settings_name = 'cut_q1'

metric_shorthand = np.array(['auc', 'lo', 'hi', 'bs', 'bss', 'bss_se', 'sens', 'tiw', 'zero'])
metric_labels = np.array(['Area under the curve (AUC)', 'Lower limit', 'Upper limit', 'Brier score', 'Brier skill score',
                          'Brier skill score standard error', 'Sensitivity', 'Time in warning', 'Zero chance percentile'])


def load_versions(version_name):

    versions = {
        'cut_q1': {'numbers':[22, 29, 17, 15], 'sample_window':10},
        'cut_q2': {'numbers':[23, 30, 18, 16], 'sample_window':10},
        'cut_q3': {'numbers':[24, 31, 19, 17], 'sample_window':10},
        'cut_q4': {'numbers':[25, 32, 20, 18], 'sample_window':10},
        'unbal_2': {'numbers':[18, 25, 13, 11], 'sample_window':2},
        'unbal_4': {'numbers':[19, 26, 14, 12], 'sample_window':4},
        'unbal_10': {'numbers':[20, 27, 15, 13], 'sample_window':10},
        'unbal_20': {'numbers':[21, 28, 16, 14], 'sample_window':20},
        'unbal_40': {'numbers':[17, 20, 12, 10], 'sample_window':40},
        'bal_2': {'numbers':[14, 17, 9, 7], 'sample_window':2},
        'bal_4': {'numbers':[15, 18, 10, 8], 'sample_window':4},
        'bal_10': {'numbers':[12, 12, 5, 3], 'sample_window':10},
        'bal_20': {'numbers':[16, 19, 11, 9], 'sample_window':20},
        'bal_40': {'numbers':[17, 20, 12, 10], 'sample_window':40},
        'first': {'numbers':[0, 8, 2, 1], 'sample_window':10},
        'remove_one': {'numbers': [0, 10, 3, 2], 'sample_window': 10},
        'all_but_one': {'numbers': [0, 12, 5, 3], 'sample_window': 10},

    }


    return versions[version_name]['numbers'], versions[version_name]['sample_window']


def load_forecast(pt, model_name):
    dir_path = '/media/projects/daniel_lstm/results_log/' + model_name
    file_path = dir_path + '/%d.pt' % pt

    y, yhat = torch.load(file_path)

    return y, yhat


def forecasts_to_results(y, yhat):
    sz_yhat, inter_yhat = met.split_yhat(y, yhat)
    a, lo, hi = met.auc_hanleyci(sz_yhat, inter_yhat)
    bs, bs_ref, bss, bss_se = met.brier_skill_score(sz_yhat, inter_yhat, p_sz=0.5)
    sen, tiw, zero_chance = met.sens_tiw(y, yhat, extrapolate=True)

    return np.array([a, lo, hi, bs[0], bss, bss_se, sen, tiw, zero_chance])


def results_all_patients(model_name):
    results = np.zeros((8, 9))
    for i, pt in enumerate([1,6,8,9,10,11,13,15]):
        # print(i, pt)
        y, yhat = load_forecast(pt, model_name)
        results[i] = forecasts_to_results(y, yhat)
    return results


def results_all_lengths(settings_name):

    version_numbers, sample_window = load_versions(settings_name)
    min_batch_size = 16 * sample_window  # for 1mBalanced: 20 samples / sz

    long_version = version_numbers[3]
    medium_version = version_numbers[2]
    short_version = version_numbers[1]
    min_version = version_numbers[0]

    results = np.zeros((4, 8, 9))
    if min_version>0:
        min_model_name = '1m_v%d_c%dp4c%dp4d%dd_%d_adam' % (min_version, min_c1, min_c2, min_fc1, min_batch_size) + str(
            learning_rate)[2:] + trans_text
        results[0] = results_all_patients(min_model_name)

    short_model_name = 'short_v%d_' % (
        short_version)  # changing the naming convention as it will get too messy, just updating version every time
    medium_model_name = 'medium_v%d_' % (medium_version)
    long_model_name = 'long_v%d_' % (long_version)

    results[1] = results_all_patients(short_model_name)
    results[2] = results_all_patients(medium_model_name)
    results[3] = results_all_patients(long_model_name)

    return results


def compare_balancing(metric_index):

    balanced_results = np.zeros((5,4,8,9))
    balanced_names = ['bal_2', 'bal_4','bal_10', 'bal_20', 'bal_40']

    unbalanced_results = np.zeros((5, 4, 8, 9))
    unbalanced_names = ['unbal_2', 'unbal_4', 'unbal_10', 'unbal_20', 'unbal_40']

    for i,name in enumerate(balanced_names):
        balanced_results[i] = results_all_lengths(name)

    for i,name in enumerate(unbalanced_names):
        unbalanced_results[i] = results_all_lengths(name)

    # bal_mean = balanced_results.mean(axis=2)
    # unbal_mean = unbalanced_results.mean(axis=2)
    #
    # bal_auc = bal_mean[:,:,0]
    # unbal_auc = unbal_mean[:,:,0]

    bal_auc = balanced_results[:,:,:,metric_index]
    unbal_auc = unbalanced_results[:, :, :, metric_index]

    bal_mean = bal_auc.mean(axis=2)
    unbal_mean = unbal_auc.mean(axis=2)


    # NOT RIGHT YET!!!!

    print(bal_mean)
    print(unbal_mean)
    # print(unbal_auc)

    # for i in range(5):
    #     b = bal_auc[i]
    #     u = unbal_auc[i]
    #     b_flat = b.flatten()
    #     u_flat = u.flatten()
    #     t, p = stats.ttest_rel(b_flat, u_flat)
    #     print(i, t, p)

    b_flat = bal_auc.flatten()
    u_flat = unbal_auc.flatten()
    t, p = stats.ttest_rel(b_flat, u_flat)
    print(t, p)

    fig, ax = plt.subplots(figsize=(7, 4))

    w = .15
    for i in range(5):
        b = bal_mean[i]
        u = unbal_mean[i]
        x = np.array([i-1.5*w, i-.5*w, i+.5*w, i+1.5*w])

        ax.plot(x, b, 'o', color=(.3, .3, .3), markersize=4, label='Balanced')

        if i<4:
            ax.plot(x, u, 'o', color=(.3, .3, .3), markersize=4, markerfacecolor=(1,1,1), label='Unbalanced')
            for j in range(4):
                dy = u[j] - b[j]
                arrow_color = 'r'
                if dy>0: arrow_color = 'g'
                ax.arrow(x=x[j], y=b[j], dx=0, dy=dy, color=arrow_color, length_includes_head=True, head_width=0)
        # plt.plot(x, u, '.r')

    ax.set_xticks([0,1,2,3,4])
    ax.set_xticklabels(['2-minute', '4-minute', '10-minute', '20-minute', '40-minute'])
    ax.set_xlabel('Sample window length')

    ax.set_ylabel(metric_labels[metric_index])

    ax.legend(['Balanced', 'Unbalanced'])

    plt.savefig('/home/daniel/Desktop/LSTM_figs/results/compare_balancing_v1_' + metric_shorthand[metric_index] + '.png',
                bbox_inches='tight', dpi=600, format='png')


def sanity_check():
    ''' Make a simple figure comparing Full LSTM with minimal LSTM

    :return:
    '''

    full_results = np.array([.62, .64, .61, .80])
    stripped_results = np.array([.50, .52, .55, .60])

    t, p = stats.ttest_rel(full_results, stripped_results)

    print('Sanity check p: ', p)


def improvement_for_1min():
    old_results = np.array([.62, .64, .61, .80])
    spread_results = np.array([.68, .73, .63, .82])
    spec_results = np.array([.71, .77, .72, .83])

    _, p_spread = stats.ttest_rel(old_results, spread_results)
    _, p_spec = stats.ttest_rel(spread_results, spec_results)

    print('Sanity check p: ', p_spread,  p_spec)
    print('old av', old_results.mean())
    print('spread av', spread_results.mean())
    print(spread_results.mean()-old_results.mean())
    print('spec av', spec_results.mean())
    print(spec_results.mean() - spread_results.mean())


def compare_min_to_LSTM(metric_index):
    # results = np.zeros((4, 3, 8, 9))
    # names = ['first', 'remove_one', 'all_but_one', 'all']
    #
    # for i, name in enumerate(names):
    #     results[i] = results_all_lengths(name)
    #
    # results = results[:, 1:]
    #
    # print(results.shape)

    results = np.array([
        [  # First
            [  # Short
                [.820, .671, .969, 0, .263, .00217, 56.25, 12.50, 50.1],
                [.640, .406, .875, 0, .135, .00293, 63.64, 18.18, 0.1],
                [.785, .704, .867, 0, .444, .00143, 68.85, 21.31, 19.7],
                [.750, .631, .870, 0, .348, .00221, 75.00, 28.13, 21.9],
                [.776, .696, .856, 0, .442, .00143, 64.62, 18.46, 1.5],
                [.893, .837, .950, 0, .701, .00080, 71.21, 10.61, 48.5],
                [.782, .711, .853, 0, .441, .00129, 64.20, 16.05, 7.4],
                [.414, .192, .636, 0, 0-.041,.00174, 53.85, 61.54, 0]
            ],
            [  # Medium
                [.803, .647, .958, 0, .399, .00234, 87.50, 31.25, 50],
                [.595, .354, .836, 0, .073, .00397, 45.46, 18.18, 18.2],
                [.770, .687, .854, 0, .401, .00154, 68.85, 24.59, 21.3],
                [.855, .760, .950, 0, .437, .00169, 65.62, 12.5, 12.5],
                [.752, .669, .836, 0, .378, .00151, 63.08, 21.5, 6.15],
                [.891, .834, .948, 0, .625, .00100, 71.21, 12.12, 6.61],
                [.770, .698, .843, 0, .423, .00131, 66.67, 17.28, 2.4],
                [.538, .313, .764, 0, .006, .00146, 46.15, 46.15, 0]
            ],
            [  # Long
                [.648, .456, .841, 0, .152, .00913, 62.50, 18.75, 6.3],
                [.550, .304, .795, .269080997, .05166, .012203, 63.636, 45.454, 0.000],
                [.711502284, .620154, .8028, .21271, .3525, .00490, 75.409, 31.147, 11.475],
                [.773437500, .658072, .8888, .20897, .2930, .00678, 62.500, 15.625, 9.3750],
                [.845443787, .777472, .9134, .16548, .4752, .00411, 70.769, 13.846, 7.692],
                [.877295684, .816855, .9377, .14194, .6245, .00302, 75.757, 15.151, 36.363],
                [.728014022, .650511, .8055, .21404, .3092, .00440, 60.493, 19.753, 9.876],
                [.541420118, .316200, .7666, .27349, .0245, .01359, 53.846, 38.461, 15.384]
            ]
        ],
        [  # remove one
            [  # Short
                [.617, .421, .814, 0, .153, .00291, .50, 25, 12.5],
                [.595, .354, .836, 0, .111, .00462, 63.64, 36.36, 0],
                [.800, .721, .878, 0, .436, .00134, 63.93, 16.39, 19.7],
                [.781, .667, .895, 0, .427, .0021,  65.63, 18.75, 3.1],
                [.836, .766, .906, 0, .513, .00119, 73.85, 18.46, 0],
                [.863, .800, .927, 0 ,.601, .00105, 63.64, 12.12, 50],
                [.810, .743, .877, 0, .451, .00116, 62.96, 9.88, 8.64],
                [.562, .338, .786, 0, .048, .00353, 46.15, 30.77, 7.7]
            ],
            [  # Medium
                [.771, .606, .937, 0, .371, .003, .50, 6.25, 50],
                [.583, 340, .825, 0, .109, .0047, 54.55, 27.27, 0],
                [.773, .69, .856, 0, .412, .00143, 65.57, 27.87, 34.4],
                [.797, .687, .907, 0, .374, .00186, 68.75, 12.5, 3.1],
                [.786, .707, .864, 0, .433, .00138, 50.77,12.31, 18.46],
                [.864, .801, .928, 0, .58, .00101, 7.79, 13.64, 39.4],
                [.724, .646, .802, 0, .357, .00133, 50.62, 6.2, 0],
                [.393, .174, .613, 0, -.988, .00344, 46.15, 61.54, 0]
            ],
            [  # Long
                [.680, .493, .867, 0, .163, .00304, 62.5, 25, 6.25],
                [.645, .410, .879,0,  .207, .00365, 63.64, 27.3, 0],
                [.747, .660, .834, 0, .373, .00158, 65.57, 21.3, 21.3],
                [.699, .571, .828, 0, .292, .002, 65.63, 28.1, 18.8],
                [.793, .716, .871, 0, .415, .00135, 73.8, 20, 4.6],
                [.877, .817, .938, 0, .625, .00101, 75.76, 15.15, 36.36],
                [.729, .651, .806, 0, .324, .00133, 59.26, 14.82, 0],
                [.491, .265, .717, 0, -.0003, .00416, 46.2, 38.46, 0]
            ]
        ],
        [  # All but one
            [  # Short
                [.756, .586, .926, 0, .294, .00285, 75, 25, 31.25],
                [.512, .266, .759, 0, -.0086, .0055, 54.55, 36.36, 9.09],
                [.84, .768, .911, 0, .511, .0012, 70.49, 16.39, 45.9],
                [.74, .618, .862, 0, .375, .0022, 62.5, 21.88, 21.88],
                [.768, .687, .849, 0, .431, .001, 73.85, 26.15, 4.6],
                [.915, .864, .965, 0, .7, .00079, 74.24, 7.6, 16.7],
                [.830, .767, .894, 0, .514, .00102, 64.2, 6.17, 1.2],
                [.494, .268, 720, 0, .024, .00389, 53.85, 38.46, 15.39]
            ],
            [  # Medium
                [.658, .467, .849, 0, .142, .00252, 62.5, 31.25, 31.25],
                [.517, .27, .763,  0 ,.056, .00615, 45.46, 27.27, 0],
                [.822, .747, .897, 0, .443, .00135, 70.49, 19.67, 9.84],
                [.744, .622, .865, 0, .311, .00192, 59.38, 21.88, 3.13],
                [.759, .577, .842, 0, .363, .00147, 66.15, 26.15, 10.77],
                [.844, .776, .911, 0, .549, .00116, 74.24, 21.21, 39.39],
                [.804, .737, .872, 0, .439, .0113, 72.84, 18.52, 9.9],
                [.642, .427, .857, 0, .147, .00204, 61.54, 30.77, 7.7]
            ],
            [  # Long
                [.721, .542, .899, 0, .109, .00165, 56.25, 12.5, 6.25],
                [.711, .491, .93, 0, .256, .00457, 3.64, 27.27, 9.09],
                [.768, .684, .852, 0, .351, .00142, 67.21, 27.869, 31.15],
                [.717, .591, .843, 0, .318, .00209, 62.5, 31.3, 31.3],
                [.735, .649, .820, 0, .321, .00162, 61.54, 21.54, 3.08],
                [.893, .836, .949, 0, .635, .00094, 71.2, 9.09, 45.46],
                [.727, .650, .805, 0, .312, .0014, 53.09, 17.28, 3.7],
                [.524, .298, .750, 0, -.02, .0030, 53.85, 30.77, 0]
            ]
        ],
        [  # All removed
            [  # Short
                [.600, .401, .798, 0, .116, .00307, 81.3, 43.8, 6.3],
                [.686, .460, .912, 0, .151, .00252, 81.8, 27.27, 36],
                [.645, .547, .743, 0, .199, .00176, 49.2, 21.31, 11.475],
                [.707, .579, .834, 0, .265, .00236, 62.5, 21.88, 3.1],
                [.628, .533, .724, 0, .162, .00194, 53.8, 26.2, 0],
                [.873, .811, .934, 0, .631, .001, 72.73, 13.64, 12.12],
                [.747, .672, .823, 0, .308, .0013, 58, 18.52, 1.24],
                [.506, .28, .732, 0, .0059, .00354, 53.85, 38.46, 0]
            ],
            [  # Medium
                [.516, .313, .719, 0, .025, .00294, 68.75, 62.5, 6.25],
                [.649, .415, .882, 0, .088, .00446, 72.73, 27.27, 09.1],
                [.719, .629, .810, 0, .242, .00139, 65.57, 31.15, 4.9],
                [.649, .515, .784, .0, .149, .0022, 62.5, 34.4, 0],
                [.661, .515, .784, 0, .2, .00164, 66.15, 40, 0],
                [.849, .782, .916, 0, .507, .00118, 69.7, 12.12, 39.4],
                [.757, .683, .831, 0, .232, .00098, 58.0, 18.52, 0],
                [.58, 357, 803, 0, .104, .0023, 76.92, 53.85, 38.46]
            ],
            [  # Long
                [.568, .368, .769, 0, .0682, .00405, 62.5, 43.75, 6.25],
                [.574, .331, .818, 0, .053, .00367, 63.64, 27.27, 0],
                [.669, .574, .765, 0, .206, .00159, 59.02, 42.62, 13.12],
                [.704, .576, .832, 0, .268, .00219, 53.1, 31.3, 6.3],
                [.573, .475, .672, 0, .099, .00181, 66.15, 52.31, 1.54],
                [.842, .773, .910, 0, .525, .00122, 66.67, 15.15, 10.61],
                [.687, .606, .769, 0, .236, .00145, 66.67, 27.16, 0],
                [.503, .277, .729, 0, -.00187, .00381, 46.15, 15.39, 0]
            ]
        ]
    ])
    print(results.shape)
    results = results[:, :, :, metric_index]
    mean = results.mean(axis=2)

    # print(results_all_patients('long_v2_'))

    fig, ax = plt.subplots(figsize=(7, 4))

    w = .25
    for i in range(4):
        y = mean[i]
        x = np.array([i - w, i, i + w])

        ax.plot(x, y, 'o', color=(.3, .3, .3), markersize=4, label='Balanced')

        # if i < 4:
        #     ax.plot(x, u, 'o', color=(.3, .3, .3), markersize=4, markerfacecolor=(1, 1, 1), label='Unbalanced')
        #     for j in range(4):
        #         dy = u[j] - b[j]
        #         arrow_color = 'r'
        #         if dy > 0: arrow_color = 'g'
        #         ax.arrow(x=x[j], y=b[j], dx=0, dy=dy, color=arrow_color, length_includes_head=True, head_width=0)
        # plt.plot(x, u, '.r')

    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['Basic', 'Remove one', 'Remove all but 1', 'Remove All'])
    # ax.set_xlabel('LSTM input')

    ax.set_ylabel(metric_labels[metric_index])

    # ax.legend(['Balanced', 'Unbalanced'])

    plt.savefig(
        '/home/daniel/Desktop/LSTM_figs/results/compare_LSTM_input_v1_' + metric_shorthand[metric_index] + '.png',
        bbox_inches='tight', dpi=600, format='png')

compare_min_to_LSTM(metric_index=0)
#
# print(results_all_patients('long_v1_'))

# 'first': {'numbers': [0, 8, 2, 1], 'sample_window': 10},
# 'remove_one': {'numbers': [0, 10, 3, 2], 'sample_window': 10},
# 'all_but_one': {'numbers': [0, 12, 5, 3], 'sample_window': 10},