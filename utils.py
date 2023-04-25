import numpy as np
from scipy.interpolate import interp1d
import scipy.signal as sg
from scipy import interpolate


#utils 

def outersum(a,b):
    return np.outer(a,np.ones_like(b))+np.outer(np.ones_like(a),b)

def Spec_Mix(x,y, gamma, mu, sigma=1):
    return sigma**2 * np.exp(-gamma*outersum(x,-y)**2)*np.cos(2*np.pi*mu*outersum(x,-y))

def Sinc(x,y, delta, mu, sigma=1):
    return sigma**2 * np.sinc(-delta*outersum(x,-y))*np.cos(2*np.pi*mu*outersum(x,-y))

def Rect(x,delta=1,mu=0,sigma=1):
    return sigma**2 / (2*delta) * np.where(np.abs(x-mu) < delta/2, np.ones_like(x), np.zeros_like(x))

def Spec_Mix_sine(x,y, gamma, mu, sigma=1):
    return sigma**2 * np.exp(-gamma*outersum(x,-y)**2)*np.sin(2*np.pi*mu*outersum(x,-y))
    
def Spec_Mix_spectral(x, y, alpha, gamma, mu, sigma=1):
    magnitude = np.pi * sigma**2 / (np.sqrt(alpha*(alpha + 2*gamma)))
    return magnitude * np.exp(-np.pi**2/(2*alpha)*outersum(x,-y)**2 - 2*np.pi*2/(alpha + 2*gamma)*(outersum(x,y)/2-mu)**2)

def freq_covariances(x, y, alpha, gamma, mu, sigma=1, kernel = 'sm'):
    if kernel == 'sm':
        N = len(x)
        #compute kernels 
        K = 1/2*(Spec_Mix_spectral(x, y, alpha, gamma, mu, sigma) + Spec_Mix_spectral(x, y, alpha, gamma, -mu, sigma))
        P = 1/2*(Spec_Mix_spectral(x, -y, alpha, gamma, mu, sigma) + Spec_Mix_spectral(x, -y, alpha, gamma, -mu, sigma))
        real_cov = 1/2*(K + P) + 1e-8*np.eye(N)
        imag_cov = 1/2*(K - P) + 1e-8*np.eye(N)
    return real_cov, imag_cov

def time_freq_SM_re(x, y, alpha, gamma, mu, sigma=1):
    at = alpha/(np.pi**2)
    gt = gamma/(np.pi**2)
    L = 1/at + 1/gt
    return (sigma**2)/(np.sqrt(np.pi*(at+gt))) * np.exp(outersum(-(x-mu)**2/(at+gt), -y**2*np.pi**2/L) ) *np.cos(-np.outer(2*np.pi*(x/at+mu/gt)/(1/at + 1/gt),y))

def time_freq_SM_im(x, y, alpha, gamma, mu, sigma=1):
    at = alpha/(np.pi**2)
    gt = gamma/(np.pi**2)
    L = 1/at + 1/gt
    return (sigma**2)/(np.sqrt(np.pi*(at+gt))) * np.exp(outersum(-(x-mu)**2/(at+gt), -y**2*np.pi**2/L) ) *np.sin(-np.outer(2*np.pi*(x/at+mu/gt)/(1/at + 1/gt),y))

def time_freq_covariances(x, t, alpha, gamma, mu, sigma, kernel = 'sm'):
    if kernel == 'sm':
        tf_real_cov = 1/2*(time_freq_SM_re(x, t, alpha, gamma, mu, sigma) + time_freq_SM_re(x, t, alpha, gamma, -mu, sigma))
        tf_imag_cov = 1/2*(time_freq_SM_im(x, t, alpha, gamma, mu, sigma) + time_freq_SM_im(x, t, alpha, gamma, -mu, sigma))
    return tf_real_cov, tf_imag_cov

def WF_loss(S1, S2, freqs1, freqs2=None):
        if freqs2 == None:
            freqs2 = freqs1

        nS1 = S1/np.sum(S1)
        nS2 = S2/np.sum(S2)

        #quantiles
        supp_quant = np.linspace(0,1, 1000)
        a1, qnS1 = inverse_histograms(nS1, freqs1, supp_quant, method='linear')
        a2, qnS2 = inverse_histograms(nS2, freqs2, supp_quant, method='linear')

        return np.sum((nS2-nS1)**2)

def find_nearest_arg(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def correct_weights(theta, freq_th, sigma_n):
    #theta = height, loc, width
    temp = theta
    for i in range(int(len(theta)/3)):
        height = theta[3*i]
        loc = theta[3*i+1]
        width = theta[3*i+2]
        if loc > freq_th:
            temp[3*i] = 0 #set height to zero
            print(f'entro: {i}')
    print(f'temp: {temp}')
    temp[0::3] = temp[0::3] / np.sqrt(np.sum(temp[0::3]**2) + sigma_n**2)  
    print(f'temp: {temp}')
    print(f'suma: {(np.sum(temp[0::3]**2) + sigma_n**2)  }')
    return temp

def FourierTransform(frequencies,times,signal, window = None):
    #times = np.concatenate((times, times+times[-1] + times[1] - times[0] , ))
    #signal = np.concatenate((signal, signal+signal[-1] + signal[1] - signal[0] , ))

    if window is not None:
        w = sg.get_window(window, len(signal))
        f = interpolate.interp1d(np.linspace(np.min(times), np.max(times), len(signal)), w)
        w = f(times)
        signal *= w

    ft = np.zeros(len(frequencies))*1j
    for t, s in zip(times, signal):
        ft += s*np.exp(-t*frequencies*1j*2*np.pi)
    return ft/len(times)

def Periodogram(frequencies,times,signal, window = None):
    F = FourierTransform(frequencies,times,signal, window)
    S = (F.real)**2+(F.imag)**2

    #step = times[1]- times[0]
    #freqs, S = sg.periodogram(signal, 1/step, window)
    #f = interpolate.interp1d(freqs, S)
    #S = f(frequencies)

    return S

def Bartlett(frequencies,times,signal,M, window = None):
    N = len(times)
    indices = np.array_split(np.arange(N), M)

    sort = np.argsort(times)
    times = times[sort]
    signal = signal[sort]

    F = FourierTransform(frequencies,times[indices[0]],signal[indices[0]], window)
    S = ((F.real)**2+(F.imag)**2)*len(indices[0])
    for m in np.arange(1,M):
        F = FourierTransform(frequencies,times[indices[m]],signal[indices[m]], window)
        S += ((F.real)**2+(F.imag)**2)*len(indices[m])
    return S/N


def Welch(frequencies,times,signal,M, window = None):
    N = len(times)
    M_new = np.int(M+1)
    indices = np.array_split(np.arange(N), M_new)

    #generate overlapping indices 
    for i in np.arange(len(indices)-1):
        indices[i] = np.concatenate((indices[i], indices[i+1]))
    #remove nonoverlaping tail
    indices = indices[:-1] 
    M_new = M_new -1

    sort = np.argsort(times)
    times = times[sort]
    signal = signal[sort]

    F = FourierTransform(frequencies,times[indices[0]],signal[indices[0]], window)
    S = ((F.real)**2+(F.imag)**2)*len(indices[0])
    total_points = len(indices[0])
    for m in np.arange(1,M_new):
        F = FourierTransform(frequencies,times[indices[m]],signal[indices[m]], window)
        S += ((F.real)**2+(F.imag)**2)*len(indices[m])
        total_points += len(indices[m])
    return S/total_points




def inverse_histograms(mu, S, Sinv, method='linear'):
    """
    Given a distribution mu compute its quantile function

    Parameters
    ----------

    mu     : histogram
    S      : support of the histogram
    Sinv   : support of the quantile function
    method : name of the interpolation method (linear, quadratic, ...)

    Returns
    -------

    cdfa   : the cumulative distribution function and
    q_Sinv : the inverse quantile function of the distribution mu

    """
    epsilon = 1e-14
    A = mu>epsilon
    A[-1] = 0
    Sa = S[A]

    cdf = np.cumsum(mu)
    cdfa = cdf[A]
    if (cdfa[-1] == 1):
        cdfa[-1] = cdfa[-1] - epsilon

    cdfa = np.append(0, cdfa)
    cdfa = np.append(cdfa, 1)

    #if S[0] < 0:
    #    # print('weird for a psd!')
    #    Sa = np.append(S[0]-1, Sa)
    #else:
    #    # set it to zero in case of PSDs
    #    Sa = np.append(0, Sa)
    
    Sa = np.append(S[0], Sa)
    Sa = np.append(Sa, S[-1])

    q = interp1d(cdfa, Sa, kind=method)
    q_Sinv = q(Sinv)
    return cdfa, q_Sinv

def rect(x, a):
    res = np.zeros(x.shape)
    res[np.abs(x) <= a/2] = 1
    return res
