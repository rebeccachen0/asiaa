import numpy as np
import scipy.optimize as op
import datetime
import emcee
import matplotlib.pyplot as plt
import corner
from scipy.stats import norm, halfnorm

def import_format_data():
    '''
    Imports HARP data, extracts dates and main beam efficiency. Removes daytime values, zeros, 
    and converts dates from YYYYMMDD to days elapsed. Creates global variables to be used later.

    Inputs: None

    Outputs: None, creates days and planets_etamb global variables
    '''
    mars = np.genfromtxt('data/harp_mars.dat', usecols=np.arange(0, 9))
    jupiter = np.genfromtxt('data/harp_jupiter.dat', usecols=np.arange(0, 8))
    uranus = np.genfromtxt('data/harp_uranus.dat', usecols=np.arange(0, 8))

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
    days = days/np.max(days)


def minor(matrix, i, j):
    '''
    This is adapted code from http://code.activestate.com/recipes/189971-basic-linear-algebra-matrix/
    Creates the minor version of a Numpy matrix. Not actually used in my implementation, 
    but may be useful in the future?

    Inputs: 
    matrix: Numpy matrix
    i, j (ints): row and column indices to remove

    Output: Numpy minor matrix
    '''
    #input should be numpy matrix
    matrix = np.asarray(matrix)
    m = np.zeros((matrix.shape[0]-1, matrix.shape[1]-1))
    m = np.matrix(m)
    # loop through the matrix, skipping over the row and column specified by i and j
    minor_row = minor_col = 0
    for row in range(matrix.shape[0]):
        if not row == i: # skip row i
            for col in range(matrix.shape[1]):
                if not col == j: # skip column j
                    m[(minor_row, minor_col)] = matrix[(row, col)]
                    minor_col += 1
            minor_col = 0
            minor_row += 1
    return m


def lnlike(theta, x, y):
    '''
    Log likelihood function. Also calculates the optimal lambda, but only prints it 
    (currently commented out).

    Inputs:
    theta: parameters
    x, y : x and y array values, in our case days and planets_etamb

    '''
    m, b, delty1, delty2, sigsq1, sigsq2, a1, a2, L = theta
    BN_lim = 200

    y_mod1 = m*x + b - delty1
    y_mod2 = m*x + b - delty2
    chisq1 = (1/sigsq1) * np.matmul(y-y_mod1, (y-y_mod1).T)
    chisq2 = (1/sigsq2) * np.matmul(y-y_mod2, (y-y_mod2).T)
    if chisq1 > BN_lim:
        chisq1 = BN_lim
    if chisq2 > BN_lim:
        chisq2 = BN_lim

    H = [np.exp(-chisq1/2), np.exp(-chisq2/2)]
    beta1 = -2.0*np.log(a1/np.sqrt(2*np.pi*sigsq1))
    beta2 = -2.0*np.log(a2/np.sqrt(2*np.pi*sigsq2))

    M = []
    for i in range(len(days)):
        if i == 0:
            temp_y = y[i+1:]
            temp_x = x[i+1:]
        else:
            temp_y = np.concatenate((y[:i],y[i+1:]))
            temp_x = np.concatenate((x[:i],x[i+1:]))
        temp_y_mod1 = m*temp_x + b - delty1
        temp_y_mod2 = m*temp_x + b - delty2
        minor_chisq1 = sigsq1 * np.matmul(temp_y-temp_y_mod1, (temp_y-temp_y_mod1).T)
        minor_chisq2 = sigsq2 * np.matmul(temp_y-temp_y_mod2, (temp_y-temp_y_mod2).T)
        M.append([np.exp(-.5*(beta1 - minor_chisq1)), np.exp(-.5*(beta2 - minor_chisq2))])
    
    MxH = np.matmul(np.matrix(M), np.matrix(H).T)
    
    #calculate optimal lambda
    full_a = np.array([[a1, a2],]*len(x))
    full_H = np.array([H,]*len(x))
    N = np.multiply(np.divide(M,full_a),full_H)
    NxH = np.matmul(np.matrix(N), np.matrix(H).T)
    lambda_star = -0.5*np.sum(np.divide(NxH,MxH))
    # print(lambda_star)

    return np.log(MxH).sum() + L*(a1+a2-1)


def lnprior(theta):
    '''
    Sets the priors. Make sure to return the sum of them all.
    '''
    m, b, delty1, delty2, sigsq1, sigsq2, a1, a2, L = theta
    if 0.0 < b < 1.0 and 0.0 < a1 < 1.0 and 0.0 < a2 < 1.0 and sigsq1 > 0.0 and sigsq2 > 0.0 and L > 0.0:
        pri_delty1 = norm.logpdf(delty1, 0.0, 0.5)
        pri_delty2 = norm.logpdf(delty2, 0.3, 0.5)
        pri_sigsq1 = norm.logpdf(sigsq1, .01, .01)
        pri_sigsq2 = norm.logpdf(sigsq2, .04, .01)
        pri_a1 = norm.logpdf(a1, 0.5, .5/3.0)
        pri_a2 = norm.logpdf(a2, 0.5, .5/3.0)
        pri_L = norm.logpdf(L, -2.04976186279e-41, 10**-5)
        return pri_delty1 + pri_delty2 + pri_sigsq1 + pri_sigsq2 + pri_a1 + pri_a2 + pri_L
    return -np.inf


def lnprob(theta, x, y):
    '''
    Defines the function to put into the MCMC sampler.
    '''
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y)


def doMCMC(theta_start, nwalkers, nsteps, walker_paths_filename, contour_filename):
    '''
    Sets up MCMC, does the sampling, generates walker paths plot and contour plot. 
    Returns the 16th, 50th, and 86th percentile results.

    Inputs:
    theta_start: array containing starting positions for the walkers 
    nwalkers (int): number of walkers
    nsteps (int): number of steps
    walker_paths_filename (string): path and filename to save walker paths plot to
    contour_filename (string): same as above

    Output: results list, ex: results[parameter number][0 for 16th percentile, 1 for 50th, 2 for 84th]
    '''
    m_start, b_start, delty1_start, delty2_start, sigsq1_start, sigsq2_start, a1_start, a2_start, L_start = theta_start
    ndim = len(theta_start)
    pos = [np.random.randn(ndim)*1e-4*[m_start, b_start, delty1_start, delty2_start, sigsq1_start, sigsq2_start, a1_start, a2_start, L_start] for i in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(days, planets_etamb), threads=4, a=2.0)
    sampler.run_mcmc(pos, nsteps)

    plt.figure(1)
    fig, axes = plt.subplots(ndim, figsize=(12, 9), sharex=True)
    samples = sampler.chain
    labels = ["m", "b", "delty1", "delty2", "sigsq1", "sigsq2", "a1", "a2", "L"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i].T, "k", alpha=0.3)
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number");
    fig.savefig(walker_paths_filename)
    plt.show()

    flat_samples = sampler.chain[:, 200:, :].reshape((-1, ndim))

    fig = corner.corner(flat_samples[:,[0,1,4,5,6,7]], labels=["$m$", "$b$", "$sigsq1$", "$sigsq2$", "$a1$", "$a2$"],
                      truths=[m_start, b_start, sigsq1_start, sigsq2_start, a1_start, a2_start])
    fig.savefig(contour_filename)

    results = []
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        results.append(mcmc)

    return results


def main():
    import_format_data()

    m_start = 0.00001
    b_start = .6
    delty1_start = 0.0
    delty2_start = 0.3
    sigsq1_start = .01
    sigsq2_start = .04
    a1_start = .5
    a2_start = .5
    L_start = 1.0

    theta_start = [m_start, b_start, delty1_start, delty2_start, sigsq1_start, sigsq2_start, a1_start, a2_start, L_start]

    nwalkers = 200
    nsteps = 1000
    walker_paths_filename = 'walkers_test.png'
    contour_filename = 'contour_test.png'
    results = doMCMC(theta_start, nwalkers, nsteps, walker_paths_filename, contour_filename)
    
    labels = ['m', 'b', 'delty1', 'delty2', 'sigsq1', 'sigsq2', 'a1', 'a2', 'L']
    for i in range(len(theta_start)):
        low = results[i][0]
        mid = results[i][1]
        high = results[i][2]
        print('{}: {} +{} -{}'.format(labels[i], mid, high-mid, mid-low))

if __name__ == '__main__':
    main()
