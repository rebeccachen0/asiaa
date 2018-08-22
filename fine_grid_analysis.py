import forward_modeling as fm
import pickle
import numpy as np

results1 = pickle.load( open( "/Volumes/drive/full_results.p", "rb" ) )
results2 = pickle.load( open( "/Volumes/drive/full_results2.p", "rb" ) )
results3 = pickle.load( open( "/Volumes/drive/full_results3.p", "rb" ) )
results4 = pickle.load( open( "/Volumes/drive/full_results4.p", "rb" ) )
results5 = pickle.load( open( "/Volumes/drive/full_results5.p", "rb" ) )
results6 = pickle.load( open( "/Volumes/drive/full_results6.p", "rb" ) )
results7 = pickle.load( open( "/Volumes/drive/full_results7.p", "rb" ) )
results8 = pickle.load( open( "/Volumes/drive/full_results8.p", "rb" ) )

print(len(results1), len(results2), len(results3), len(results4), len(results5), len(results6), len(results7), len(results8))

def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result


full_results_concat = merge_dicts(results1, results2, results3, results4, results5, results6, results7, results8)
print("length of concat:", len(full_results_concat))

#calculate MADM median absolute deviation median
all_ks_stats=[]
for param, test in full_results_concat.items():
    all_ks_stats.append(test[1][0])
print("length of all stats:", len(all_ks_stats))
tenth_percent = int(len(all_ks_stats)/100)
top_pointone_percent = np.sort(all_ks_stats)[:tenth_percent]
top_median = np.median(top_pointone_percent)
deviations = []
for stat in top_pointone_percent:
    deviations.append(top_median - stat)

sorted_deviations = np.sort(deviations)
MADM = np.median(np.absolute(deviations))    
print("MADM:" , MADM)
fifth = np.percentile(sorted_deviations, 5)
ninety_fifth = np.percentile(sorted_deviations, 95)
print("5th and 95th percentile:", fifth, ninety_fifth)
print("{}+{}-{}".format(MADM, np.absolute(MADM-ninety_fifth), np.absolute(MADM-fifth)))
