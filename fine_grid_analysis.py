import forward_modeling as fm
import pickle
import numpy as np
import ast
from scipy.stats import norm
from operator import itemgetter

#load in full_results from forward_modeling script

#load fine grid slices
# results1 = pickle.load( open( "/Users/jonty/Desktop/full_results.p", "rb" ) )
# results2 = pickle.load( open( "/Users/jonty/Desktop/full_results2.p", "rb" ) )
# results3 = pickle.load( open( "/Users/jonty/Desktop/full_results3.p", "rb" ) )
# results4 = pickle.load( open( "/Users/jonty/Desktop/full_results4.p", "rb" ) )
# results5 = pickle.load( open( "/Users/jonty/Desktop/full_results5.p", "rb" ) )
# results6 = pickle.load( open( "/Users/jonty/Desktop/full_results6.p", "rb" ) )
# results7 = pickle.load( open( "/Users/jonty/Desktop/full_results7.p", "rb" ) )
# results8 = pickle.load( open( "/Users/jonty/Desktop/full_results8.p", "rb" ) )
# full_results_concat = merge_dicts(results1, results2, results3, results4, results5, results6, results7, results8)

#loads coarse grid
full_results_concat = pickle.load( open( "/Users/rebeccachen/Desktop/asiaa/project/final_coarse_grid.p", "rb" ) )



def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


#calculate MADM median absolute deviation median
all_ks_stats=[]
for param, tests in full_results_concat.items():
    mu = norm.fit(tests[1][0])
    all_ks_stats.append(mu)
print("length of all stats:", len(all_ks_stats))
tenth_percent = int(len(all_ks_stats)/100)
top_pointone_percent = np.sort(all_ks_stats)[:tenth_percent]
top_median = np.median(top_pointone_percent)
deviations = []
for stat in top_pointone_percent:
    deviations.append(top_median - stat)

sorted_deviations = np.sort(deviations)
k = 1.4826
MADM = np.median(np.absolute(deviations))    
uncert = MADM*k
print("MADM:" , MADM)
print("uncertainty:", uncert)

# same as full_results, but only the mean fitted from a Gaussian
reduced_full_results = {}
for params, arr in full_results_concat.items():
	mu = norm.fit(arr[1][0])[0]
	print(mu)
	reduced_full_results[params] = mu

sort = sorted(reduced_full_results.items(), key=lambda x:x[1])

#obtain parameters that fall within range of MADM
top_m =[]
top_b = []
top_bias_sig = []
top_rel = []
for param_stat in sort:
	param = param_stat[0]
	param = param[1:-1]
	params = [float(i) for i in param.split()]
	m, b, bias_sig, rel = params
	stat = param_stat[1]
	print(stat)
	if stat <= MADM + uncert:
		top_m.append(m)
		top_b.append(b)
		top_bias_sig.append(bias_sig)
		top_rel.append(rel)

#print results
mu_m, std_m = norm.fit(top_m)
print("m mu, std:", mu_m, std_m)
mu_b, std_b = norm.fit(top_b)
print("b mu, std:", mu_b, std_b)
mu_bias_sig, std_bias_sig = norm.fit(top_bias_sig)
print("bias_sig mu, std:", mu_bias_sig, std_bias_sig)
mu_rel, std_rel = norm.fit(top_rel)
print("rel mu, std:", mu_rel, std_rel)

lowest_ks, lowest_params = fm.minimize_stats(full_results_concat, 1)
print("lowest KS and params: ", lowest_ks, lowest_params)
