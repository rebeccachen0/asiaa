import numpy as np
import scipy.optimize as op
import datetime
import emcee
import matplotlib.pyplot as plt
import corner
from scipy.stats import norm

mars = np.genfromtxt('harp_mars.dat', usecols=np.arange(0, 9))
jupiter = np.genfromtxt('harp_jupiter.dat', usecols=np.arange(0, 8))
uranus = np.genfromtxt('harp_uranus.dat', usecols=np.arange(0, 8))

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

planets_etamb = np.concatenate((mars_etamb, jupiter_etamb, uranus_etamb), axis=0)
raw_dates = np.concatenate((mars_dates, jupiter_dates, uranus_dates), axis=0)
dates = [datetime.datetime.strptime(str(int(date)),'%Y%m%d') for date in raw_dates]

oldest = min(dates)
days = np.array([np.float64((date - oldest).days) + 1 for date in dates])

m_true = 0.0
b_true = .5

lnyerr_true = -2.3
lnV_true = -1.6

def lnlike(theta, x, y):
    m, b, lnyerr, lnV = theta
    #print(m, b, lnyerr, lnV)

    term1 = 0
    term2 = 0
    for i in range(len(x)):
        vhat = np.matrix([[-m], [1]])/(np.sqrt(1 + m**2))
        Z = np.matrix([[x[i]], [y[i]]])
        delt = np.matmul(vhat.T, Z) - b * np.cos(np.arctan(m))
        S = np.matrix([[0, 0], [0, np.exp(lnyerr)**2]])
        bsig2 = np.matmul(np.matmul(vhat.T, S), vhat)
        term1 += -0.5 * np.log(bsig2 + np.exp(lnV))
        term2 += -0.5 * (delt**2)/(bsig2 + np.exp(lnV))
    return (term1 + term2)


nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, [m_true, b_true, lnyerr_true, lnV_true], args=(days, planets_etamb))
m_ml, b_ml, lnyerr_ml, lnV_ml = result["x"]

def lnprior(theta):
    m, b, lnyerr, lnV = theta
    if 0.0 < b < 1.0:
#   if -1.0 < m < 1.0 and 0.0 < b < 1.0 and -5.0 < lnyerr < 0.0 and -3.0 < lnV < 0.0:
        pri_m = norm.logpdf(m, 0.0, 1.0)
        pri_lnyerr = norm.logpdf(lnyerr, -2.5, 2.5)
        pri_lnV = norm.logpdf(lnV, -1.5, 1.5)
        return pri_m + pri_lnyerr + pri_lnV
    return -np.inf


def lnprob(theta, x, y):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y)


ndim, nwalkers = 4, 200
pos = [result["x"] + np.random.randn(ndim)*[.2, .1, .5, .3] for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(days, planets_etamb), threads=4)
sampler.run_mcmc(pos, 500)


plt.figure(1)
fig, axes = plt.subplots(4, figsize=(10, 7), sharex=True)
samples = sampler.chain
labels = ["m", "b", "lnyerr", "lnV"]
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");
plt.show()

flat_samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
fig = corner.corner(flat_samples, labels=["$m$", "$b$", "$lnyerr$", "$lnV$"],
                      truths=[m_true, b_true, lnyerr_true, lnV_true])
fig.savefig("uncert_scatter_informpri.png")

print("Mean acceptance fraction: {0:.3f}"
                .format(np.mean(sampler.acceptance_fraction)))