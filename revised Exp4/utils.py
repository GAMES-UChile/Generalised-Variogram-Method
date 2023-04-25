import numpy as np
from scipy.interpolate import interp1d
import scipy.signal as sg
from scipy import interpolate
import matplotlib.pyplot as plt



#utils 

def outersum(a,b):
    return np.outer(a,np.ones_like(b))+np.outer(np.ones_like(a),b)

def Spec_Mix(x,y, gamma, mu, sigma=1):
    return sigma**2 * np.exp(-gamma*outersum(x,-y)**2)*np.cos(2*np.pi*mu*outersum(x,-y))

def FSpec_Mix(freqs, gamma, mu, sigma=1):
    return sigma**2 * np.sqrt(np.pi/gamma) * np.exp(-(np.pi*(freqs-mu))**2/gamma)

def SE(x,y, sigma, gamma):
    return sigma**2 * np.exp(-gamma*outersum(x,-y)**2)

def WN(x,y, sigma):
    o = outersum(x,-y)**2
    return 1.0*sigma**2 * (o==0)

def ARD(X1, X2, lscale, base_l, magnitude):
    #M3 = Lambda.squeeze()*Lambda + L_inv**2  # Use broadcasting to avoid transpose
    M3 = lscale@lscale.T + np.diag(base_l**2)
    d = X1[:,None] - X2[None,...]            #  Use broadcasting to avoid loops
    # order=F for memory layout (as your arrays are (N,M,D) instead of (D,N,M))
    return magnitude * np.exp(-0.5 * np.einsum("ijk,kl,ijl->ij", d, M3, d, order = 'F'))

def WN_MD(X1, X2, std_var):
    d = X1[:,None] - X2[None,...]            #  Use broadcasting to avoid loops
    # order=F for memory layout (as your arrays are (N,M,D) instead of (D,N,M))
    o = np.einsum("ijk,kl,ijl->ij", d, np.eye(d.shape[-1]), d, order = 'F')
    return 1.0*std_var**2 * (o==0)

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
    arg = np.argsort(times)
    times = times[arg]
    signal = signal[arg]

    if window is not None:
        w = sg.get_window(window, len(signal))
        f = interpolate.interp1d(np.linspace(np.min(times), np.max(times), len(signal)), w)
        w = f(times)
        signal *= w

    ft = np.zeros(len(frequencies))*1j
    dt = np.diff(times)
    dt = np.append(dt,dt[-1])
    for t, s, d in zip(times, signal, dt):
        ft += d*s*np.exp(-t*frequencies*1j*2*np.pi)
    return ft/np.sqrt(np.sum(dt))


def Periodogram(frequencies,times,signal, window = None):
    F = FourierTransform(frequencies,times.flatten(),signal, window)
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


def plot_mogptk(model):
    time_train, y_train = model.dataset[0].get_train_data()
    time_test, y_test = model.dataset[0].get_test_data()
    time = model.dataset[0].get_prediction_x()
    mean, lower, upper = model.predict(sigma=2, predict_y=False)
    time_train = time_train[0] 
    time_test = time_test[0]
    time = time[0]
    mean = mean[0]
    lower = lower[0]
    upper = upper[0]    
    idx = np.argsort(time)
    
    #posterior moments for time
    plt.figure(figsize=(18,6))
    plt.plot(time_train,y_train,'.r', markersize=10, label='observations')
    plt.plot(time_test,y_test,'.g', markersize=10, label='latent')
    plt.plot(time[idx], mean[idx], color='blue', ls = 'dotted', label='posterior mean')
    plt.fill_between(time[idx], lower[idx], upper[idx], color='blue',alpha=0.3, label='95% error bars')

    if hasattr(model, 'losses'):
        plt.title(f'Observations, latent values and GP (NLL: {model.losses[-1]})')
    plt.legend()
    plt.xlim([min(time),max(time)])
    plt.tight_layout()


def envelope(sig, distance):
    # split signal into negative and positive parts
    u_x = np.where(sig > 0)[0]
    l_x = np.where(sig < 0)[0]
    u_y = sig.copy()
    u_y[l_x] = 0
    l_y = -sig.copy()
    l_y[u_x] = 0
    
    # find upper and lower peaks
    u_peaks, _ = sg.find_peaks(u_y, distance=distance)
    l_peaks, _ = sg.find_peaks(l_y, distance=distance)
    
    # use peaks and peak values to make envelope
    u_x = u_peaks
    u_y = sig[u_peaks]
    l_x = l_peaks
    l_y = sig[l_peaks]
    
    # add start and end of signal to allow proper indexing
    end = len(sig)
    u_x = np.concatenate((u_x, [0, end]))
    u_y = np.concatenate((u_y, [0, 0]))
    l_x = np.concatenate((l_x, [0, end]))
    l_y = np.concatenate((l_y, [0, 0]))
    
    # create envelope functions
    u = interpolate.interp1d(u_x, u_y)
    l = interpolate.interp1d(l_x, l_y)
    return u, l
