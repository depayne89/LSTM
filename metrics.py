import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, kendalltau, rankdata, sem
import math
import sys

def accuracy(y,yhat):
    correct = (yhat == y).sum()
    accuracy = correct / len(y)
    print('Accuracy', accuracy)


def split_yhat(y, yhat):
    # print('yhat min', yhat.min())
    sz_yhat = yhat[y==1]
    inter_yhat = yhat[y==0]
    # print('Split yhat', sz_yhat, inter_yhat)
    return sz_yhat, inter_yhat


def auc(sz, inter, plot=False):
    ''' Calculates the Area under the curve of the receiver operating characteristic curve

    :param sz: array, forecasts for seizure samples
    :param inter: array, forecasts for interictal samples
    :param plot: boolean, turn on to plot ROC curve
    :return: auc, area under the curve
    '''

    if sz.size==0 or inter.size==0:
        return 0., 0, 0

    sz[sz<10**(-40)] = 10**(-40)
    inter[inter < 10 ** (-40)] = 10 ** (-40)
    # Initialise graph and fpr, tpr arrays.
    minimum = min(sz.min(), inter.min())  # smallest forecast (to get minimum of x-axis)
    # print('Min', minimum)
    min_exp = int(np.log10(minimum))-1  # smallest forecast in log scale
    steps_per_decade = 20  # resolution of the AUC calculation. Here decade refers to order of magnitude
    vals = -min_exp*steps_per_decade+1
    fpr = np.empty(vals)
    tpr = np.empty(vals)

    # Slowly increase threshold, determining fpr and tpr at each iteration
    for i, threshold_log in enumerate(np.arange(min_exp, .1/steps_per_decade,1./steps_per_decade)):  # 2nd argument is exclusive limit (makes inclusive limit = 0 ie 10**0=1. second argument is step size
        threshold = 10**threshold_log
        fpr[i] = float(inter[inter>threshold].size) / inter.size
        tpr[i] = float(sz[sz>threshold].size) / sz.size

    # AUC calculated as area under curve with the curve extrapolated between points using trapezoids
    auc = np.trapz(np.flip(tpr,0), np.flip(fpr, 0)) # Flip  reverses the order to make auc positive

    # Plot ROC curve
    if plot:
        plt.figure(3)
        plt.plot(fpr, tpr, linewidth=1.5)
        plt.xlabel('False Positive Rate', fontsize=6)
        plt.ylabel('True Positive Rate', fontsize=6)
        plt.title('')
        # plt.legend()


    return auc, fpr, tpr


def auc_se(a, m, n):
    ''' Calculates the standard error of an AUC value

    :param a: area under the ROC curve
    :param m: int, number of sz samples
    :param n: int, number of interictal samples
    :return:
    '''

    q1 = a/(2-a)                                                    # intermediate step
    q2 = (2*a**2) / (1+a)                                           # intermediate step
    a_var = (a*(1-a) + (m-1)*(q1-a**2) + (n-1)*(q2-a**2))/(m*n)     # variance of AUC
    a_se = math.sqrt(a_var)                                         # standard error of AUC

    return a_se


def auc_hanleyci(sz, inter, alpha=.05, bonferroni=1, plot=False):
    ''' Calculates AUC with confidence interval using the Hanley Method

    :param sz: array, forecasts for seizure samples
    :param inter: array, forecasts for interictal samples
    :param alpha: p-value significance threshold, default= 0.05
    :param bonferroni: int, number of AUCs calculated. Used to make the bonferroni adjustment for multiple tests
    :param plot: boolean, turn on to plot ROC curve
    :return:
    '''

    # ------- Calculating AUC --------
    a, _, _ = auc(sz, inter, plot)

    # -------- Calculating CI --------
    m = sz.size  # Sz samples
    n = inter.size  # Non sz samples
    alpha_adjusted = alpha / bonferroni  # alpha with Bonfferoni adjustment
    z = norm.ppf(1 - alpha_adjusted / 2)  # z-score of CI edge
    a_se = auc_se(a, m, n)  # standard error of AUC
    ci_low = a - z * a_se  # confidence interval minimum
    ci_hi = a + z * a_se  # confidence interval maximum

    # Set limits of CI to limits of AUC
    if ci_low < 0:
        ci_low = 0
    if ci_hi > 1:
        ci_hi = 1

    ci = [ci_low, ci_hi]  # Confidence interval

    return a, ci_low, ci_hi


def shuffley(y):
    len = y.size
    out = y[np.random.randint(0, len, len)]
    return out


def auc_on_shuffled(n, sz, inter, alpha=.05, bonferroni=1):
    sz_len = sz.size
    combo = np.concatenate((sz, inter))
    aucs = np.zeros(n)
    for i in range(n):
        np.random.shuffle(combo)
        sz_shuffled = combo[:sz_len]
        inter_shuffled = combo[sz_len:]
        a, _, _ = auc(sz_shuffled, inter_shuffled)
        aucs[i] = a

    auc_mean = aucs.mean()

    se = aucs.std() / math.sqrt(n)

    return auc_mean, se


def brier_score(sz_forecasts, inter_forecasts, p_sz):

    n_samples = sz_forecasts.size + inter_forecasts.size

    p_sz = float(sz_forecasts.size) / float(sz_forecasts.size + inter_forecasts.size)

    # print 'p_sz'
    # print p_sz
    # print np.mean(sz_forecasts)
    # print np.mean(inter_forecasts)


    sum = 0
    for sample in sz_forecasts:
        sum += (sample - 1.)**2
    for sample in inter_forecasts:
        sum += (sample - 0.)**2

    return sum / n_samples


def brier_skill_score(sz_forecasts, inter_forecasts, p_sz):

    bs = brier_score(sz_forecasts, inter_forecasts, p_sz)

    # All forecasts the same
    # sz_forecasts_r = np.ones(sz_forecasts.shape)*p_sz
    # inter_forecasts_r = np.ones(inter_forecasts.shape)*p_sz

    # Forecasts shuffled
    forecasts = np.concatenate((sz_forecasts, inter_forecasts))

    bss_array = np.empty(1000)
    for i in range(1000):
        np.random.shuffle(forecasts)
        sz_forecasts_r = forecasts[:sz_forecasts.size]
        inter_forecasts_r = forecasts[sz_forecasts.size:]

        # print sz_forecasts[:10]
        # print sz_forecasts_r[:10]

        bs_ref = brier_score(sz_forecasts_r, inter_forecasts_r, p_sz)

        bss_array[i] = 1 - bs/bs_ref

    bss = np.mean(bss_array)
    bss_se = sem(bss_array)

    return bs, bs_ref, bss, bss_se