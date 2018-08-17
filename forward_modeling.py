import numpy as np
import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm, halfnorm, kstest, ks_2samp, wilcoxon, mannwhitneyu, probplot, expon
from scipy.optimize import curve_fit
import pickle

def import_format_data():
    mars = np.genfromtxt('harp_mars.dat', usecols=np.arange(0, 9))
    jupiter = np.genfromtxt('harp_jupiter.dat', usecols=np.arange(0, 8))
    uranus = np.genfromtxt('harp_uranus.dat', usecols=np.arange(0, 8))

    mars_unc = np.genfromtxt('harp_mars_unc.dat')
    jupiter_unc = np.genfromtxt('harp_jupiter_unc.dat')
    uranus_unc = np.genfromtxt('harp_uranus_unc.dat')
    planets_unc = np.concatenate((mars_unc[:,9], jupiter_unc[:,8], uranus_unc[:,8]), axis=0)
    global yerr
    yerr = np.mean(planets_unc[planets_unc!=0])

    mars_etamb = mars[:,5]
    jupiter_etamb = jupiter[:,5]
    uranus_etamb = uranus[:,5]

    mars_dates = mars[:,0]
    jupiter_dates = jupiter[:,0]
    uranus_dates = uranus[:,0]

    mars = mars[mars[:,1] > 5]
    mars = mars[mars[:,1] < 19]
    mars = mars[mars[:,5] > 0]
    jupiter = jupiter[jupiter[:,1] > 5]
    jupiter = jupiter[jupiter[:,1] < 19]
    jupiter = jupiter[jupiter[:,5] > 0]
    uranus = uranus[uranus[:,1] > 5]
    uranus = uranus[uranus[:,1] < 19]
    uranus = uranus[uranus[:,5] > 0]

    global planets_etamb
    planets_etamb = np.concatenate((mars_etamb, jupiter_etamb, uranus_etamb), axis=0)
    raw_dates = np.concatenate((mars_dates, jupiter_dates, uranus_dates), axis=0)
    dates = [datetime.datetime.strptime(str(int(date)),'%Y%m%d') for date in raw_dates]
    oldest = min(dates)
    global days
    days = np.array([np.float64((date - oldest).days) + 1 for date in dates])


def draw_samp_yerrfix(x, m, b, yerr, bias_mu, bias_sig, rel):
	'''
	Draws a single sample from a model with set y error, exponentially modeled bias, and a bias weight term.

	Inputs: 

	Outputs: an array of sampled y values
	'''
	result = []
	bias = expon.rvs(bias_mu, bias_sig, size = len(x))
	result = yerr - rel*bias + (m*x + b)
	return np.array(result)


def calc1000tests_yerrfix(m, b, yerr, bias_mu, bias_sig, rel):
    ks_stats = [] #each has 1000 test stats, corresponding to one set of params
    mwu_stats = []
    wilcoxon_stats = []
    ks_pv = [] #each has 1000 p-values, corresponding to one set of params
    mwu_pv = []
    wilcoxon_pv = []
    samples = []
    for i in range(1000):
        trial = draw_samp_yerrfix(days, m, b, yerr, bias_mu, bias_sig, rel)
        samples.append(trial)
        stat = ks_2samp(planets_etamb, trial)
        ks_stats.append(stat[0])
        ks_pv.append(stat[1])
        stat = mannwhitneyu(planets_etamb, trial, alternative = 'two-sided')
        mwu_stats.append(stat[0])
        mwu_pv.append(stat[1])
        stat = wilcoxon(planets_etamb, trial)
        wilcoxon_stats.append(stat[0])
        wilcoxon_pv.append(stat[1])
        
    return samples, ks_stats, mwu_stats, wilcoxon_stats, ks_pv, mwu_pv, wilcoxon_pv


def generate_samples_stats(m1, m2, b1, b2, bias_sig1, bias_sig2, rel1, rel2, num_trials):
    '''
	Inputs: (m1, m2) : lower and upper range for m
	(b1, b2) : same as above
	(bias_sig1, bias_sig2) : same as above
	(rel1, rel2) : same as above
	num_trials(int): specify how many values of each parameter you want to test
	e.g. num_trials=5 with 4 parameters gives a total of 4^5 parameter combinations

	Outputs: full_results(dict): a dictionary with key-value pairs (string of parameters tested,
		list of format [samples, [stats_ks, pvals_ks], [stats_mwu, pvals_mwu], [stats_w, pvals_w]])
	'''
    m = np.linspace(m1, m2, num=num_trials)
    b = np.linspace(b1, b2, num=num_trials)
    bias_sig = np.linspace(bias_sig1, bias_sig2, num=num_trials)
    rel = np.linspace(rel1, rel2, num=num_trials)

	#generate all test statistics, version without yerr and with rel
    a, B, c, d = np.meshgrid(m, b, bias_sig, rel)
    arr = np.stack((np.ravel(a), np.ravel(B), np.ravel(c), np.ravel(d)), axis=-1)

    full_results = {}
    for i in arr:
        samples, stats_ks, stats_mwu, stats_w, pvals_ks, pvals_mwu, pvals_w = calc1000tests_yerrfix(i[0], i[1], yerr, 0.0, i[2], i[3]) #each an array of 1000 pvals
        full_results[str(i)] = [samples, [stats_ks, pvals_ks], [stats_mwu, pvals_mwu], [stats_w, pvals_w]]
    return full_results


def minimize_stats(full_results):
    '''
    Inputs: the dictionary return result of generate_samples_stats()

    Outputs:
    '''
	#find lowest peak of histogram for each test
    lowest_ks = np.inf
    lowest_mwu = np.inf
    lowest_w = np.inf
    stat_num = 0 #0-test stats 1-p-values
    for params, arr in full_results.items():
        mu_ks, std_ks = norm.fit(arr[1][stat_num])
        mu_mwu, std_mwu = norm.fit(arr[2][stat_num])
        mu_w, std_w = norm.fit(arr[3][stat_num])
        curr_samples = arr[0] #array of 1000 arrays of 550 values, 1000 sampled datasets
        if mu_ks != 0 and mu_ks < lowest_ks:
            lowest_ks = mu_ks
            lowest_ks_std = std_ks
            lowest_params_ks = params
            p_vals_ks = arr[1][1]
            stats_ks = arr[1][0]
            samples_ks = curr_samples
        if mu_mwu != 0 and mu_mwu < lowest_mwu:
            lowest_mwu = mu_mwu
            lowest_mwu_std = std_mwu
            lowest_params_mwu = params
            p_vals_mwu = arr[2][1]
            stats_mwu = arr[2][0]
            samples_mwu = curr_samples
        if mu_w != 0 and mu_w < lowest_w: # looking at lowest mu that still passes a threshold
            lowest_w = mu_w
            lowest_w_std = std_w
            lowest_params_w = params
            p_vals_w = arr[3][1]
            stats_w = arr[3][0]
            samples_w = curr_samples
    return lowest_ks, lowest_params_ks


def main():
    import_format_data()
    print("Testing models, generating stats")
    full_results = generate_samples_stats(-.001, .001,.4, 1.0,.1, 1.0,.01, 2.0, 2)
    pickle.dump(full_results, open("full_results.p", "wb"))
    lowest, lowest_params = minimize_stats(full_results)
    print("Lowest ks test stat and params: ", lowest, lowest_params)

    arr = pickle.load(open("full_results.p", "rb"))
    print("Checking if it saves and loads:")
    for key, value in arr.items():
        print(key)





if __name__ == '__main__':
    main()

