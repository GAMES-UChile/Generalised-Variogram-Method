import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from scipy.interpolate import interp1d
import time
import scipy.signal as sg
from scipy import special
from utils import *


plot_params = {'legend.fontsize': 18,
          'figure.figsize': (15, 5),
         'xtick.labelsize':'18',
         'ytick.labelsize':'18',
         'axes.titlesize':'24',
         'axes.labelsize':'22'}
plt.rcParams.update(plot_params)

class gpinit:

    # Class Attribute none yet

    # Initializer / Instance Attributes
    def __init__(self, space_input=None, space_output=None, time_test = None, y_test = None, aim=None, kernel = 'SE', order = 1):

        if aim is None:
            self.offset = np.median(space_input)
            self.x = space_input - self.offset
            self.y = space_output
            self.post_mean = None
            self.post_cov = None
            self.post_mean_r = None
            self.post_cov_r = None
            self.post_mean_i = None
            self.post_cov_i = None
            self.time_label = None
            self.signal_label = None
            self.initialise_params()
        elif aim == 'sampling':
            self.kernel = kernel
            #self.sigma = 1
            #self.gamma = 1/2
            #self.mu = 1e-10
            #self.sigma_n = 0
            self.x = space_input
            if self.Mx==1:
                self.x = np.sort(self.x,axis=0)
                self.time = np.linspace(np.min(self.x), np.max(self.x), 500)
            self.order = order
        elif aim == 'regression':
            self.x = space_input
            self.y = space_output
            self.Nx = len(self.x)
            self.time_label = None
            self.signal_label = None
            self.initialise_params()
        elif aim == 'learning':
            self.bias = np.mean(space_output)
            self.y = space_output - self.bias
            self.x = space_input
            self.Nx= space_input.shape[0]
            self.Mx= space_input.shape[1]
            if self.x is None: 
                self.x = np.arange(self.Nx)
            self.theta = None
            self.fixed_freqs = False
            self.freqs = None
            self.S = None
            self.kernel = kernel
            self.sigma_n = 0
            self.w_noise = False
            if np.var(np.diff(self.x),axis=0).all() < 1e-5:
                self.uniform_sampling = True
            else: 
                self.uniform_sampling = False
            self.real_world = False
            self.time_test = time_test
            self.y_test = y_test
            if self.Mx==1 and time_test is not None: #if input dims == 1, sort times
                self.time = np.sort(np.concatenate((space_input, time_test)))
                self.time = np.sort(self.time.flatten())
            self.order = order
            

    def initialise_params(self):
        self.Nx = len(self.x)
        self.alpha = 1/2/((np.max(self.x)-np.min(self.x))/2)**2
        self.sigma = np.std(self.y)
        self.gamma = 1/2/((np.max(self.x)-np.min(self.x))/self.Nx)**2
        self.mu = 0.05
        self.sigma_n = np.std(self.y)/10
        self.time = np.linspace(np.min(self.x), np.max(self.x), 500)
        self.w = np.linspace(0, self.Nx/(np.max(self.x)-np.min(self.x))/16, 500)

    def set_freqs (self, freqs):
        self.freqs = freqs
        self.freq_step = self.freqs[1]-self.freqs[0]
        self.fixed_freqs = True

    def reload(self, space_input, space_output):
        self.x = space_input
        self.y = space_output - self.bias
        self.Nx = len(self.x)



    def neg_log_likelihood_old(self):        
        Y = self.y
        Gram = Spec_Mix(self.x,self.x,self.gamma,self.mu,self.sigma) + 1e-8*np.eye(self.Nx)
        K = Gram + self.sigma_n**2*np.eye(self.Nx)
        (sign, logdet) = np.linalg.slogdet(K)
        return 0.5*( Y.T@np.linalg.solve(K,Y) + logdet + self.Nx*np.log(2*np.pi))


    def nlogp(self, hypers):

        Y = self.y
        if self.kernel == 'SE':
            hypers = np.exp(hypers)
            sigma = hypers[0]
            gamma = hypers[1]
            sigma_n = hypers[2]
            Gram = SE(self.x, self.x, sigma, gamma)
        
        elif self.kernel == 'SM':
            sigma = np.exp(hypers[0])
            gamma = np.exp(hypers[1])
            mu = np.exp(hypers[2])
            sigma_n = np.exp(hypers[3])
                
            Gram = Spec_Mix(self.x,self.x,gamma,mu,sigma)
        

        K = Gram + sigma_n**2*np.eye(self.Nx) + 1e-5*np.eye(self.Nx)
        (sign, logdet) = np.linalg.slogdet(K)
        return 0.5*( Y.T@np.linalg.solve(K,Y) + logdet + self.Nx*np.log(2*np.pi))

    def dnlogp(self, hypers):
        
        Y = self.y
        if self.kernel == 'SE':
            hypers = np.exp(hypers)
            sigma = hypers[0]
            gamma = hypers[1]
            sigma_n = hypers[2]

            Gram = SE(self.x, self.x, sigma, gamma)
            K = Gram + sigma_n**2*np.eye(self.Nx) + 1e-5*np.eye(self.Nx)
            h = np.linalg.solve(K,Y).T

            dKdsigma = 2*Gram/sigma
            dKdgamma = -Gram*(outersum(self.x,-self.x)**2)
            dKdsigma_n = 2*sigma_n*np.eye(self.Nx)

            H = (np.outer(h,h) - np.linalg.inv(K))
            dlogp_dsigma = sigma * 0.5*np.trace(H@dKdsigma)
            dlogp_dgamma = gamma * 0.5*np.trace(H@dKdgamma)
            dlogp_dsigma_n = sigma_n * 0.5*np.trace(H@dKdsigma_n)

            return np.array([-dlogp_dsigma, -dlogp_dgamma, -dlogp_dsigma_n])
        
        elif self.kernel == 'SM':

            sigma = np.exp(hypers[0])
            gamma = np.exp(hypers[1])
            mu = np.exp(hypers[2])
            sigma_n = np.exp(hypers[3])

            
            Gram = Spec_Mix(self.x,self.x,gamma,mu,sigma)
            K = Gram + sigma_n**2*np.eye(self.Nx) + 1e-5*np.eye(self.Nx)
            h = np.linalg.solve(K,Y).T

            dKdsigma = 2*Gram/sigma
            dKdgamma = -Gram*(outersum(self.x,-self.x)**2)
            dKdmu = -2*np.pi*Spec_Mix_sine(self.x,self.x, gamma, mu, sigma)*outersum(self.x,-self.x)
            dKdsigma_n = 2*sigma_n*np.eye(self.Nx)

            H = (np.outer(h,h) - np.linalg.inv(K))
            dlogp_dsigma = sigma * 0.5*np.trace(H@dKdsigma)
            dlogp_dgamma = gamma * 0.5*np.trace(H@dKdgamma)
            dlogp_dmu = mu * 0.5*np.trace(H@dKdmu)
            dlogp_dsigma_n = sigma_n * 0.5*np.trace(H@dKdsigma_n)
            return np.array([-dlogp_dsigma, -dlogp_dgamma, -dlogp_dmu, -dlogp_dsigma_n])

    def train(self, flag = 'quiet'):

        #hypers0 = np.array([np.log(self.sigma), np.log(self.gamma), np.log(self.mu), np.log(self.sigma_n)])

        res = minimize(self.nlogp, np.log(self.theta), args=(), method='L-BFGS-B', jac = self.dnlogp, options={'maxiter': 500, 'disp': False})
        #self.sigma = np.exp(res.x[0])
        #self.gamma = np.exp(res.x[1])
        #self.mu = np.exp(res.x[2])
        #self.sigma_n = np.exp(res.x[3])
        #self.theta = np.array([self.mu, self.gamma, self.sigma_n ])
        self.theta = np.exp(res.x)
        if flag != 'quiet':
            print('Hyperparameters are:')
            print(f'sigma ={self.sigma}')
            print(f'gamma ={self.gamma}')
            print(f'mu ={self.mu}')
            print(f'sigma_n ={self.sigma_n}')

    def loss(self, S1, S2, metric):
        nS1 = S1/np.sum(S1)
        nS2 = S2/np.sum(S2)

        #quantiles
        supp_quant = np.linspace(0,1, 1000)
        a1, qnS1 = inverse_histograms(nS1, self.freqs, supp_quant, method='linear')
        a2, qnS2 = inverse_histograms(nS2, self.freqs, supp_quant, method='linear')

        if metric == 'W2':
            return np.sqrt(np.sum((qnS2-qnS1)**2))
        elif metric == 'L2':
            return np.sqrt(np.sum((nS2-nS1)**2))
        elif metric == 'L1':
            return np.sqrt(np.sum(np.abs(nS2-nS1)))
        elif metric == 'W1':  
            return np.sum(np.abs(np.cumsum(nS1)-np.cumsum(nS2)))
        elif metric == 'KL':  
            return np.sum((np.log(nS1)-np.log(nS2))*nS1)
        elif metric == 'IS':  
            d = np.log(nS1)-np.log(nS2)
            return np.sum( np.exp(d) - d - 1)
            #return np.sum(nS1/nS2 - np.log(nS1) + np.log(nS1) -1  )

    def loss_SM(self, params, method):
        #params = sigma, location, scale, etc
        if np.max(params) > 1e2:
            S = np.ones_like(self.freqs)
        else: 
            params = np.exp(params)
            order = self.order
            S = np.zeros_like(self.freqs)
            for o in range(order):
                #print(params[3*o:3*o+3])
                if params[3*o+1] > self.freqs[-1]:
                    params[3*o+1] = self.freqs[-1]
                S += params[3*o]**2 * np.exp(-(self.freqs-params[3*o+1])**2/2/params[3*o+2]**2)

            if len(params) % 3 == 1:
                #print('noise in kernel')
                S += params[-1] #noise

            #S = S/np.sum(S)
            #S += self.sigma_n**2

        return self.loss(self.S, S, method)





    def loss_Dirac(self, params, method):
        #params = sigma, location, scale, etc
        S = self.sigma_n**2*np.ones_like(self.freqs)
        if np.max(params) > 1e2:
            S = self.sigma_n**2*np.ones_like(self.freqs)
        else: 
            params = np.exp(params)
            order = int(len(params)/2)
            for o in range(order):
                freq = params[2*o]
                weight = params[2*o+1]
                if freq > self.freqs[-1]:
                    freq = self.freqs[-1]
                idx = find_nearest_arg(self.freqs, freq) 
                S[idx] = weight**2

            #if len(params) % 2 == 1:
            #    #print('noise in kernel')
            #    S += params[-1] #noise

            #S = S/np.sum(S)
            #S += self.sigma_n**2

        return self.loss(self.S, S, method)


    def loss_qsinc(self, params, method):
        #params = height, location, scale, etc
        params = np.exp(params)
        order = int(len(params)/3)
        S = np.zeros_like(self.freqs)


        for o in range(order):
            S += params[3*o]**2 * np.where(np.abs(self.freqs-params[3*o+1]) < params[3*o+2]/2, np.ones_like(self.freqs), np.zeros_like(self.freqs))


        if len(params) % 3 == 1:
            S += params[-1] #noise


        return self.loss(self.S, S,method)


    


    def compute_S(self, method = 'periodogram', nbins = 3, window = None):

        if self.fixed_freqs is False:
            self.freqs = np.linspace(0,0.5*(self.Nx-1)/(np.max(self.x)-np.min(self.x)),10000)

            # method is the number of bins, 1= periodogram, <1 =bartlet, >-1 = welch
            #periodogram
            if method == 'periodogram': 
                self.method_label = f'Periodogram (w: {window})'
                S = Periodogram(self.freqs, self.x, self.y, window)

            #bartlett
            elif method == 'bartlett':
                self.method_label = f'Bartlett (w: {window}, bins: {nbins})'
                S = Bartlett(self.freqs, self.x, self.y, nbins, window)

            #welch
            elif method == 'welch':
                self.method_label = f'Welch (w: {window}, bins: {nbins})'

                S = Welch(self.freqs, self.x, self.y, nbins, window)


            if self.w_noise:
                nS = S/np.sum(S)
                cnS = np.cumsum(nS)
                self.sigma_n = np.sqrt(np.mean(nS[nS < 0.1]))

            if not self.uniform_sampling:
                S = np.where(S > 0.05*(np.max(S)-np.min(S)), S, np.zeros_like(S))
                nS = S/np.sum(S)
                cnS = np.cumsum(nS)
            
            nS = S/np.sum(S)
            cnS = np.cumsum(nS)
            if self.real_world: 
                self.freqs = self.freqs[cnS < 0.95]
            else:
                self.freqs = self.freqs[cnS < 0.99]


            self.freqs = np.linspace(0,self.freqs[-1],2000)
        
        self.freq_step = self.freqs[1]-self.freqs[0]

        if method == 'periodogram': 
            self.method_label = f'Periodogram (w: {window})'
            S = Periodogram(self.freqs, self.x, self.y, window)

        #bartlett
        elif method == 'bartlett':
            self.method_label = f'Bartlett (w: {window}, bins: {nbins})'
            S = Bartlett(self.freqs, self.x, self.y, nbins, window)

        #welch
        elif method == 'welch':
            self.method_label = f'Welch (w: {window}, bins: {nbins})'

            S = Welch(self.freqs, self.x, self.y, nbins, window)

        #lomb-scargle
        elif method == 'lomb-scargle':
            self.method_label = f'Lomb-Scarge'

            S = sg.lombscargle(self.x, self.y, self.freqs*2*np.pi)



        #if (not self.uniform_sampling) and (method != 'lomb-scargle'):
        #    S = np.where(S > 0.01*(np.max(S)-np.min(S)), S, np.zeros_like(S))


        #double sided
        if self.kernel == 'SE' or self.kernel == 'c-sinc'  :
            self.freqs = np.concatenate((-np.flipud(self.freqs),self.freqs[1:]));
            S = np.concatenate((np.flipud(S),S[1:]));


            
        #self.S = S/np.sum(S)
        self.S = S

    def general_loss(self, K, S, metric):

        #print(K.flatten().shape)
        #print(self.acf_values.flatten().shape)
        #print(self.acf_magnitudes.shape)
        if metric == 'L1':
            return np.sum(np.abs(K.flatten()-self.acf_values)*np.linspace(1,0,len(K)))
            #return np.sum(np.abs(K.flatten()-self.acf_values)/(self.acf_times+1))
        elif metric == 'L2_MD':
            d = (K.flatten()-self.acf_values.flatten())
            d *= np.linspace(1,0,len(K))**(self.Mx)#/2.0)
            #return d.T@(d/(1+self.acf_times))
            return d.T@d
            #return np.sum( (np.abs(K.flatten()-self.acf_values.flatten())/(self.acf_magnitudes +1))**2 )
        elif metric == 'L2':
            #return np.sum( (np.abs(K.flatten()-self.acf_values)/(1+self.acf_magnitudes))**2 )
            return np.sum( (np.abs(K.flatten()-self.acf_values.flatten())*np.linspace(1,0,len(K)))**2 )
        if metric == 'FL1':
            return np.sum(np.abs(S.flatten()-self.S))
        elif metric == 'FL2':
            return np.sum((S.flatten()-self.S)**2)
        elif metric == 'W2': #not ready
            return np.sqrt(np.sum((qnS2-qnS1)**2))
        elif metric == 'W1': #not ready
            return np.sum(np.abs(np.cumsum(nS1)-np.cumsum(nS2)))
        elif metric == 'KL': #not ready
            return np.sum((np.log(nS1)-np.log(nS2))*nS1)
        elif metric == 'IS': #not ready
            d = np.log(nS1)-np.log(nS2)
            return np.sum( np.exp(d) - d - 1)
            #return np.sum(nS1/nS2 - np.log(nS1) + np.log(nS1) -1  )


    def pre_loss(self, params, metric):
        #params = sigma, location, scale, etc
        if self.kernel == 'SE': 
            sigma = np.exp(params[0])
            gamma = np.exp(params[1])
            sigma_n = np.exp(params[2])
            #define K como fn de params cn noise
            K = SE(self.acf_times,0, sigma, gamma) + WN(self.acf_times,0, sigma_n)
            S = sigma**2*np.sqrt(np.pi/gamma)*SE(self.freqs,0, 1, np.pi**2/gamma) + sigma_n**2
        elif self.kernel == 'qSM': 
            sigma_n = np.exp(params[-1])
            K = WN(self.acf_times,0, sigma_n)
            S = sigma_n**2 * np.ones((len(self.freqs)))
            #print(K.shape)
            #print(S.shape)

            for i in range(self.order):
                sigma = np.exp(params[3*i+0])
                gamma = np.exp(params[3*i+1])
                mu = np.exp(params[3*i+2])
                #define K como fn de params cn noise
                #SS = FSpec_Mix(self.freqs, gamma, mu, sigma) 
                #print(SS.shape)
                K += Spec_Mix(self.acf_times,0, gamma, mu, sigma) 
                S += FSpec_Mix(self.freqs, gamma, mu, sigma) 

        elif self.kernel == 'ARD': 
            magnitude = np.exp(params[0])
            base_lengthscale = np.exp(params[1:self.Mx+1])
            lengthscale = np.exp(np.reshape(params[self.Mx+1:(self.order+1)*self.Mx +1],(self.Mx,self.order)))
            noise_stdvar = np.exp(params[-1])
            K = ARD(np.zeros((1,self.Mx)),self.acf_times.reshape(-1,self.Mx), lengthscale,base_lengthscale,magnitude)
            K += WN_MD(np.zeros((1,self.Mx)),self.acf_times.reshape(-1,self.Mx), noise_stdvar)
            #print(f'this is K: {K}')
            #print(f' dim 1: {np.zeros((1,self.Mx)).shape}')
            #print(f' dim 2: {self.acf_times.reshape(-1,self.Mx).shape}')

            #print(f'WN: {WN_MD(np.zeros((1,self.Mx)),self.acf_times.reshape(-1,self.Mx), noise_magnitude)}')
            S = None

        elif self.kernel == 'SE_MD': 
            sigma = np.exp(params[0])
            gamma = np.exp(params[1])
            noise_stdvar = np.exp(params[2])
            K = SE(0, self.acf_times, sigma, gamma)
            self.Ktheta = K
            K += WN(0,self.acf_times, noise_stdvar)
            #print(f'this is K: {K}')
            #print(f' dim 1: {np.zeros((1,self.Mx)).shape}')
            #print(f' dim 2: {self.acf_times.reshape(-1,self.Mx).shape}')

            #print(f'WN: {WN_MD(np.zeros((1,self.Mx)),self.acf_times.reshape(-1,self.Mx), noise_magnitude)}')
            S = None

        return self.general_loss(K, S, metric)

    def compute_statistics(self):
        if self.Mx == 1:
            self.compute_acf()
            self.S = Periodogram(self.freqs, self.x.flatten(), self.y)
        else:
            self.compute_acf_MD()

    def compute_KS(self):


        if self.kernel == 'SE': 
            sigma = self.theta[0]
            gamma = self.theta[1]
            sigma_n = self.theta[2]
            #define K como fn de params cn noise
            self.Ktheta = SE(self.acf_times,0, sigma, gamma) + WN(self.acf_times,0, sigma_n)
            self.Stheta = sigma**2*np.sqrt(np.pi/gamma)*SE(self.freqs,0, 1, np.pi**2/gamma) + sigma_n**2

        elif self.kernel == 'qSM': 
            sigma_n = self.theta[-1]
            K = WN(self.acf_times,0, sigma_n)
            S = sigma_n**2 * np.ones((len(self.freqs)))

            for i in range(self.order):
                sigma = self.theta[3*i+0]
                gamma = self.theta[3*i+1]
                mu = self.theta[3*i+2]
                K += Spec_Mix(self.acf_times,0, gamma, mu, sigma) 
                S += FSpec_Mix(self.freqs, gamma, mu, sigma) 

            self.Ktheta = K
            self.Stheta = S

    def initialise(self, method = 'temporal', metric = 'L2', window = None, nbins = 3, hypers0 = None, w_noise = False, verbose = False):

        


        if self.kernel == 'SE':
            self.compute_statistics()
            if hypers0 is None:
                hypers0 = np.array([1,1,1])

            if metric == 'FW2': #this is a closed-form case # repair!!

                nS = self.S/np.sum(self.S)

                #quantile
                supp_quant = np.linspace(0,1, 1000)
                aa, qnS = inverse_histograms(nS, self.freqs, supp_quant, method='linear')

                #Wasserstein
                inv_erf = special.erfinv(2*supp_quant-1)
                inv_erf[0] = 0
                inv_erf[-1] = 0
                line = supp_quant-1/2

                l_star = (np.sqrt(2)*np.sum(inv_erf*qnS)) / (2*np.sum(inv_erf**2))
                self.theta[1] = 1/(2*(l_star**2))

            else:

                hypers0 = np.log(hypers0)
                res = minimize(self.pre_loss, hypers0, args=(metric), method='Powell', options={'maxiter': 1500, 'disp': False});
                self.theta = np.exp(res.x)
                if np.var(self.y) > self.theta[0]**2 + self.theta[2]**2:
                    #print('corrected')
                    self.theta[2] = np.sqrt(np.var(self.y)-self.theta[0]**2)

        elif self.kernel == 'qSM':
            self.compute_statistics()
            #if hypers0 is None:
            #    hypers0 = 0.75 + np.random.rand(3*self.order + 1)/2 #uniformly distributed between 0.75 and 1.25

            if hypers0 is None:
                #hypers = height, loc, width 
                peaks, properties = sg.find_peaks(self.S, width=2)
                if len(peaks) < self.order:
                    order = len(peaks)
                    print(f'order greater that found peaks, order reduced to: {order}')
                arg = np.argsort(-self.S[peaks])
                peaks = peaks[arg]
                widths = properties["widths"][arg]
                #print(peaks)
                hypers0 = np.zeros(3*self.order)
                for o in range(self.order):
                    
                    hypers0[3*o+1] = 2*(widths[o]*self.freq_step)**2 #gamma 
                    hypers0[3*o] = np.sqrt(self.S[peaks[o]]) * (np.pi*hypers0[3*o+1])**0.25 #sigma
                    hypers0[3*o+2] = self.freqs[peaks[o]] #mu
                    
            hypers0_ = np.ones(3*self.order + 1)
            hypers0_[:-1] = np.log(hypers0)
            res = minimize(self.pre_loss, hypers0_, args=(metric), method='Powell', options={'maxiter': 1500, 'disp': False});
            self.theta = np.exp(res.x)
            #print(f'theta: {self.theta}')
            #print(f'theta truncated: {self.theta[0::3]}')
            if np.var(self.y) > np.sum(self.theta[0::3]**2):
                #print('corrected')
                self.theta[-1] = np.sqrt(np.var(self.y)-np.sum(self.theta[0:-1:3]**2))

        elif self.kernel == 'ARD':
            self.compute_acf2()
            if hypers0 is None:
                #print('define hypers')
                hypers0 = 0.8 + 0.4*np.random.rand(self.Mx*(self.order+1)+2) #initialise params around 1

            hypers0 = np.log(hypers0)
            res = minimize(self.pre_loss, hypers0, args=(metric), method='L-BFGS-B', options={'maxiter': 1500, 'disp': False});
            self.theta = np.exp(res.x)
            if np.var(self.y) > self.theta[0] + self.theta[-1]**2:
                #print('corrected')
                self.theta[-1] = np.sqrt(np.var(self.y)-self.theta[0])

        elif self.kernel == 'SE_MD':
            self.compute_acf2()
            if hypers0 is None:
                #print('define hypers')
                hypers0 = 0.8 + 0.4*np.random.rand(self.Mx*(self.order+1)+2) #initialise params around 1

            hypers0 = np.log(hypers0)
            res = minimize(self.pre_loss, hypers0, args=(metric), method='L-BFGS-B', options={'maxiter': 1500, 'disp': False});
            self.theta = np.exp(res.x)
            if np.var(self.y) > self.theta[0]**2 + self.theta[-1]**2:
                #print('corrected')
                self.theta[-1] = np.sqrt(np.var(self.y)-self.theta[0]**2)
        
    def train_WL(self, method = 'periodogram', metric = 'L2', window = None, nbins = 3, hypers0 = None, order = 1, w_noise = False, verbose = False):

        self.power = np.std(self.y)**2
        self.compute_S(method, nbins, window)
        self.order = order
        
        #if self.fixed_freqs == True and self.w_noise:
        if True:
            q = np.quantile(self.S, 0.01)
            #self.sigma_n = np.sqrt(np.mean(self.S[self.S < q]))
            self.sigma_n = np.sqrt(q)



        if self.kernel == 'qSM':
            if hypers0 is None:
                #hypers = height, loc, width
                peaks, properties = sg.find_peaks(self.S, width=2)
                if len(peaks) < order:
                    order = len(peaks)
                    print(f'order greater that found peaks, order reduced to: {order}')
                arg = np.argsort(-self.S[peaks])
                peaks = peaks[arg]
                widths = properties["widths"][arg]
                #print(peaks)
                hypers0 = np.zeros(3*order)
                for o in range(order):
                    hypers0[3*o] = np.sqrt(self.S[peaks[o]])
                    hypers0[3*o+1] = self.freqs[peaks[o]] 
                    hypers0[3*o+2] = widths[o]*self.freq_step
                    
            hypers0 = np.log(hypers0)
            res = minimize(self.loss_SM, hypers0, args=(metric), method='L-BFGS-B', options={'maxiter': 1500, 'disp': False});
            print(f'{metric}-ok')
            theta = np.exp(res.x)
     
            theta[0::3] = np.sqrt(self.power) * theta[0::3] / np.sqrt(np.sum(theta[0::3]**2) + self.sigma_n**2)  

        if self.kernel == 'qDirac':
            if hypers0 is None:
                #hypers = height, loc, width
                #peaks, properties = sg.find_peaks(self.S, width=2)
                peaks, properties = sg.find_peaks(self.S, width=5)

                if len(peaks) < order:
                    order = len(peaks)
                    print(f'order greater that found peaks, order reduced to: {order}')
                arg = np.argsort(-self.S[peaks])
                peaks = peaks[arg]
                #widths = properties["widths"][arg]
                #print(peaks)
                hypers0 = np.zeros(2*order)
                for o in range(order):
                    hypers0[2*o] = self.freqs[peaks[o]] 
                    hypers0[2*o+1] = np.sqrt(self.S[peaks[o]])
                    #hypers0[3*o+2] = widths[o]*self.freq_step
                    print(f'peak:{hypers0[2*o]},weight:{hypers0[2*o+1]}')
                    
            #hypers0 = np.log(hypers0)
            #res = minimize(self.loss_Dirac, hypers0, args=(metric), method='Powell', options={'maxiter': 1500, 'disp': True});
            #print(f'{metric}-ok')
            #theta = np.exp(res.x)
            #no training:
            theta = hypers0

            theta[0::3] = theta[0::3] / np.sqrt(np.sum(theta[0::3]**2) + self.sigma_n**2)  


        if self.kernel == 'qsinc':
            if hypers0 is None:
                peaks, properties = sg.find_peaks(self.S, width=5)
                if len(peaks) < order:
                    order = len(peaks)
                    print(f'order greater that found peaks, order reduced to: {order}')
                arg = np.argsort(-self.S[peaks])
                peaks = peaks[arg]
                widths = properties["widths"][arg]
                #print(peaks)
                hypers0 = np.zeros(3*order)
                for o in range(order):
                    hypers0[3*o] = np.sqrt(self.S[peaks[o]])
                    hypers0[3*o+1] = self.freqs[peaks[o]] 
                    hypers0[3*o+2] = widths[o]*self.freq_step
            
            hypers0 = np.log(hypers0)
            res = minimize(self.loss_qsinc, hypers0, args=(metric), method='Powell', options={'maxiter': 1500, 'disp': False});
            print(f'{metric}-ok')


            theta = np.exp(res.x)

        else: #closed form

            S = self.S

            #normalisation
            nS = S/np.sum(S)

            #quantile
            supp_quant = np.linspace(0,1, 1000)
            aa, qnS = inverse_histograms(nS, self.freqs, supp_quant, method='linear')

            #Wasserstein
            inv_erf = special.erfinv(2*supp_quant-1)
            inv_erf[0] = 0
            inv_erf[-1] = 0

            line = supp_quant-1/2


            if self.kernel == 'SE':
                l_star = (np.sqrt(2)*np.sum(inv_erf*qnS)) / (2*np.sum(inv_erf**2))
                theta = (0,l_star)

            elif self.kernel == 'SM':
                mu_star = np.mean(qnS)
                l_star = (np.sqrt(2)*np.sum(inv_erf*(qnS-mu_star))) / (2*np.sum(inv_erf**2))
                theta = (mu_star,l_star)

            if self.kernel == 'c-sinc':
                l_star = (np.sum(line*qnS)) / (np.sum(line**2))
                theta = (0,l_star)

            elif self.kernel == 'sinc':
                mu_star = np.mean(qnS)
                l_star = np.sum(line*(qnS-mu_star)) / (np.sum(line**2))
                theta = (mu_star,l_star)


        if verbose:
            print(f'parameters are {theta}')
            print(f'nbins are  {nbins}')
            print(f'using method {method}')
            print(f'using kernel {self.kernel}')
            print(f'frequency grids uses {len(self.freqs)} points from 0 to {self.freqs[-1]}')

        self.theta = theta
        

    def neg_log_likelihood(self):        
        Y = self.y
        if self.kernel == 'SE':
            sigma = self.theta[0]
            gamma = self.theta[1]
            sigma_n = self.theta[2]
            Gram = SE(self.x, self.x, sigma, gamma) + 1e-8*np.eye(self.Nx)
        elif self.kernel == 'SM':
            mu = self.theta[0]
            gamma = 1/2/self.theta[1]**2
            Gram = Spec_Mix(self.x, self.x, gamma, mu) + 1e-8*np.eye(self.Nx)
        elif self.kernel == 'sinc' or self.kernel == 'c-sinc':
            print('sinc?')
            mu = self.theta[0]
            delta = self.theta[1]
            Gram = Sinc(self.x,self.x,delta,mu) + 1e-8*np.eye(self.Nx)
        
        K = Gram + sigma_n**2*np.eye(self.Nx)
        (sign, logdet) = np.linalg.slogdet(K)
        return 0.5*( Y.T@np.linalg.solve(K,Y) + logdet + self.Nx*np.log(2*np.pi))

    def print_time_params(self):
        print(f'angle is: {self.theta[0]}')
        print(f'scale is: {1/2/np.pi/self.theta[1]}')



    def plot_psd(self,title = None, f_true=None, S_true=None, label_true=None):
        step = self.freqs[1]-self.freqs[0]
        if self.kernel == 'SE' or  self.kernel == 'SM':
            learnt_S = np.exp(-(self.freqs-self.theta[0])**2/(2*self.theta[1]**2))
            learnt_S /= np.sum(learnt_S)*step
            #learnt_S_label = f'{self.kernel} kernel (loc = {self.theta[0]:.4f}, scale = {self.theta[1]:.4f})'
            learnt_S_label = f'Learnt kernel'

        elif self.kernel == 'c-sinc' or self.kernel == 'sinc':
            learnt_S = np.where(np.abs(self.freqs-self.theta[0]) < self.theta[1]/2, np.ones_like(self.freqs), np.zeros_like(self.freqs))
            learnt_S /= np.sum(learnt_S)*step
            #learnt_S_label = f'Rectangle (loc = {self.theta[0]:.4f}, scale = {self.theta[1]:.4f})'
            learnt_S_label = f'Learnt kernel'

        elif self.kernel == 'qSM':
            params = self.theta
            order = int(len(params)/3)
            learnt_S = np.zeros_like(self.freqs)
            for o in range(order):
                learnt_S += params[3*o]**2 * np.exp(-(self.freqs-params[3*o+1])**2/2/params[3*o+2]**2)

            if len(params) % 3 == 1:
                learnt_S += params[-1] #noise

            learnt_S /= np.sum(learnt_S)*step
            learnt_S_label = f'Learnt {order}-comp Spectral Mix.'

        elif self.kernel == 'qDirac':
            params = self.theta
            order = int(len(params)/2)
            learnt_S = np.zeros_like(self.freqs)
            for o in range(order):
                freq = params[2*o]
                weight = params[2*o+1]
                if freq > self.freqs[-1]:
                    freq = self.freqs[-1]
                idx = find_nearest_arg(self.freqs, freq) 
                learnt_S[idx] = weight**2

                
            #if len(params) % 3 == 1:
            #    learnt_S += params[-1] #noise

            learnt_S /= np.max(learnt_S)*np.max(self.S)
            learnt_S_label = f'Learnt {order}-comp Dirac Mix.'




        elif self.kernel == 'qsinc':
            params = self.theta
            order = int(len(params)/3)
            learnt_S = np.zeros_like(self.freqs)

            for o in range(order):
                learnt_S += params[3*o]**2 * np.where(np.abs(self.freqs-params[3*o+1]) < params[3*o+2]/2, np.ones_like(self.freqs), np.zeros_like(self.freqs))

            if len(params) % 3 == 1:
                learnt_S += params[-1] #noise

            learnt_S /= np.sum(learnt_S)*step
            learnt_S_label = f'Learnt {order}-comp Sinc Mix.'




        S = self.S/(np.sum(self.S)*step)


        c = plt.cm.Set1
        plt.figure(figsize=(12*0.9,4*0.9))
        plt.plot(self.freqs,S, color = c(1) , markersize=10, label=self.method_label)
        plt.plot(self.freqs,learnt_S,color = c(0), linewidth=3, label= learnt_S_label)
        if f_true is not None:
            plt.plot(f_true, S_true,color = 'k', linewidth=3, label= label_true)
        plt.title(title)
        plt.xlabel('frequencies')
        plt.legend(loc = 0)
        #plt.xlim([min(self.x),max(self.x)])
        plt.tight_layout()


    def plot_time(self):
        plt.figure(figsize=(18,6))
        plt.plot(self.x,self.y,'.r', markersize=10, label='data')
        plt.title(f'Observations, latent values and GP (NLL: {self.neg_log_likelihood})')
        #plt.xlabel(self.time_label)
        #plt.ylabel(self.signal_label)
        plt.legend()
        #plt.xlim([min(self.x),max(self.x)])
        plt.tight_layout()
        


    def sample(self, space_input=None):

        if space_input is None: 
            self.Nx = 100
            self.x = np.random.random(self.Nx)
        elif np.size(space_input) == 1: 
            self.Nx = space_input
            self.x = np.random.random(self.Nx)
        else:
            self.x = space_input
            self.Nx = len(space_input)
        self.x = np.sort(self.x)
        if self.kernel == 'SE':
            sigma = self.theta[0]
            gamma = self.theta[1]
            sigma_n = self.theta[2]
            cov_space = SE(self.x,self.x,sigma,gamma) + sigma_n**2*np.eye(self.Nx)
        elif self.kernel == 'SM':
            cov_space = Spec_Mix(self.x,self.x,self.gamma,self.mu,self.sigma) + self.sigma_n**2*np.eye(self.Nx)
        elif self.kernel == 'sinc':
            cov_space = Sinc(self.x,self.x,self.delta,self.mu,self.sigma) + self.sigma_n**2*np.eye(self.Nx)
        elif self.kernel == 'qSM':
            sigma_n = self.theta[-1]
            cov_space = sigma_n**2*np.eye(self.Nx)
            for i in range(self.order):
                sigma = self.theta[3*i+0]
                gamma = self.theta[3*i+1]
                mu = self.theta[3*i+2]
                cov_space += Spec_Mix(self.x, self.x, gamma, mu, sigma) 


        self.y =  np.random.multivariate_normal(np.zeros_like(self.x), cov_space)

        return self.y

    def compute_acf(self):

        #calculate
        times = outersum(self.x,-self.x)
        corrs = np.outer(self.y,self.y)
        
        #reshape into vector
        times = np.reshape(times, self.Nx**2)
        corrs = np.reshape(corrs, self.Nx**2)

        #remove half
        corrs = corrs[times>=0]
        times = times[times>=0]
        th = 0.2*np.max(times)
        corrs = corrs[times < th]
        times = times[times < th]

        times = np.around((times-np.min(times))/np.max(times), 3)*np.max(times) + np.min(times)

        #unique & average
        lags, ind, inv_ind, counts = np.unique(times, return_index=True, return_inverse=True, return_counts=True, axis=None)
        average = np.zeros_like(lags)
        for i in range(len(lags)):
            average[i] = np.mean(corrs[times==lags[i]])
        
        #sample
        #idx = np.random.choice(np.arange(len(lags)), self.Nx, replace=False, p=counts/np.sum(counts))
        #lags = lags[idx]
        #average = average[idx]

        #sort
        #ind = np.unravel_index(np.argsort(lags, axis=None), lags.shape)
        ind = np.argsort(lags)
        self.acf_times = lags[ind]
        self.acf_values = average[ind]
        self.acf_magnitudes = np.abs(lags[ind])

    def compute_acf2(self):

        #calculate
        # compute autocorr function for MD inputs
        # X is a matrix of NxM: N observed vector of dimension M
        # Y is a vector of N scalar observations
        # return: MD lags and corrs
        X = self.x
        Y = self.y
        N = X.shape[0]
        M = X.shape[1]
        XT = np.tile(X,(N,1)) # repeat entire matrix downwards N times
        XR = np.repeat(X,N, axis=0) #repeat each row downwards N times
        lags = XT-XR

        YT = np.tile(Y.reshape(N,1),(N,1)) # repeat entire matrix downwards N times
        YR = np.repeat(Y.reshape(N,1),N, axis=0) #repeat each row downwards N times
        corrs = YT*YR

        #remove half
        corrs = corrs[lags[:,0]>=0]
        lags = lags[lags[:,0]>=0,:]

        #sort (ignore full lags from now on in the MD case)
        lags_m = np.sqrt(np.sum(np.abs(lags)**2,axis=1))

        th = 0.25*np.max(lags_m)
        corrs = corrs[lags_m < th]
        lags_m = lags_m[lags_m < th]


        ind = np.argsort(lags_m)
        corrs = corrs[ind]
        lags_m = lags_m[ind]
        #print(lags_m.flatten())
        #print(corrs.flatten())


        #plt.plot(lags_m, corrs)
        #quantise
        lags_m = np.around(lags_m/np.max(lags_m), 2)*np.max(lags_m)

        #unique & average
        unique_lm, ind, inv_ind, counts = np.unique(lags_m, return_index=True, return_inverse=True, return_counts=True, axis=None)
        average = np.zeros_like(unique_lm)
        for i in range(len(unique_lm)):
            average[i] = np.mean(corrs[lags_m==unique_lm[i]])
        
        #sample
        #idx = np.random.choice(np.arange(len(lags)), self.Nx, replace=False, p=counts/np.sum(counts))
        #lags = lags[idx]
        #average = average[idx]

        #sort
        #ind = np.unravel_index(np.argsort(lags, axis=None), lags.shape)
        ind = np.argsort(unique_lm)
        self.acf_times = unique_lm[ind]
        self.acf_values = average[ind]
        #self.acf_magnitudes = np.abs(lags[ind])

    def compute_acf_MD(self):
        # compute autocorr function for MD inputs
        # X is a matrix of NxM: N observed vector of dimension M
        # Y is a vector of N scalar observations
        # return: MD lags and corrs
        X = self.x
        Y = self.y
        N = X.shape[0]
        M = X.shape[1]
        XT = np.tile(X,(N,1)) # repeat entire matrix downwards N times
        XR = np.repeat(X,N, axis=0) #repeat each row downwards N times
        lags = XT-XR

        
        
        YT = np.tile(Y.reshape(N,1),(N,1)) # repeat entire matrix downwards N times
        YR = np.repeat(Y.reshape(N,1),N, axis=0) #repeat each row downwards N times
        corrs = YT*YR





        
        #remove half
        corrs = corrs[lags[:,0]>0]
        lags = lags[lags[:,0]>0,:]

        

        #magnitudes = np.sum(np.abs(lags),axis=1)
        #mm = np.max(magnitudes)
        #th = 1*mm
        #corrs = corrs[magnitudes < th]
        #lags = lags[magnitudes < th]
        #magnitudes = np.sum(np.abs(lags),axis=1)
        
        magnitudes = np.sum(np.abs(lags),axis=1)
        ind = np.argsort(magnitudes)

        corrs = corrs[ind[:40000]]
        lags = lags[ind[:40000],:]
        magnitudes = magnitudes[ind[:40000]]


        #print(corrs)
        #print(lags)

        
        #quantise
        #magnitudes = np.sum(np.abs(lags),axis=1)
        #mm = np.max(magnitudes)
        #lags = np.around(lags/mm, 1)*mm
        
        
        #unique & average
        #lags_u = np.unique(lags, axis=0)
        #n_lags = lags_u.shape[0]
        #average = np.zeros(n_lags)
        #for i in range(n_lags):
        #    dlag = np.sum(np.abs(lags-lags_u[i,:]),axis=1)
        #    average[i] = np.mean(corrs[dlag == 0])
        
        
        self.acf_values = corrs
        self.acf_magnitudes = magnitudes
        self.acf_times = lags



    def plot_acf(self,instruction):

        self.compute_acf()

        if instruction == 'plot.':
            plt.plot(self.acf_times, self.acf_values,'.', alpha=0.015)
        if instruction == 'plot-':
            plt.plot(self.acf_times, self.acf_values)
        plt.title('ACF')
        




    def compute_moments_time(self):

        #posterior moments for time
        if self.kernel == 'SE':
            sigma = self.theta[0]
            gamma = self.theta[1]
            sigma_n = self.theta[2]
            cov_space = SE(self.x,self.x, sigma, gamma) + 1e-5*np.eye(self.Nx) + WN(self.x,self.x, sigma_n) 
            cov_time = SE(self.time,self.time, sigma, gamma)
            cov_star = SE(self.time,self.x, sigma, gamma)


        elif self.kernel == 'SM':
            cov_space = Spec_Mix(self.x,self.x,self.gamma,self.mu,self.sigma) + 1e-5*np.eye(self.Nx) + self.sigma_n**2*np.eye(self.Nx)
            cov_time = Spec_Mix(self.time,self.time, self.gamma, self.mu, self.sigma)
            cov_star = Spec_Mix(self.time,self.x, self.gamma, self.mu, self.sigma)

        elif self.kernel == 'qSM':
            #params = sigma, location, scale, etc
            params = self.theta
            order = self.order
            cov_space  = np.zeros((self.Nx,self.Nx))
            cov_time = np.zeros((len(self.time),len(self.time)))
            cov_star = np.zeros((len(self.time),self.Nx))

            for o in range(order):
                sigma = params[3*o]
                mu = params[3*o+1]
                l = params[3*o+2]
                sigma_t = sigma * np.sqrt(l) * (2*np.pi)**(1/4) 
                mu_t = mu
                gamma_t = 2 * l**2 * np.pi**2
                cov_space  += Spec_Mix(self.x,self.x,gamma_t, mu_t, sigma_t) + 1e-5*np.eye(self.Nx)
                cov_time   += Spec_Mix(self.time,self.time, gamma_t, mu_t, sigma_t)
                cov_star   += Spec_Mix(self.time,self.x, gamma_t, mu_t, sigma_t)

            #if len(params) % 3 == 1:
            #    print(f'an {params[-1]}')
            #    cov_space += np.eye(self.Nx)*params[-1] #noise





        self.post_mean = np.squeeze(cov_star@np.linalg.solve(cov_space,self.y))
        self.post_cov = cov_time - (cov_star@np.linalg.solve(cov_space,cov_star.T))


    def show_hypers(self):
        print(f'gamma: {self.gamma}')
        print(f'sigma: {self.sigma}')
        print(f'sigma_n: {self.sigma_n}')
        print(f'mu: {self.mu}')

    def plot_time_posterior(self, x_true = None, y_true = None, label_true = None, flag = 'latent'):

        #posterior moments for time
        plt.figure(figsize=(18,6))
        plt.plot(self.x,self.bias + self.y,'.r', markersize=10, label='observations')
        plt.plot(self.time_test,self.y_test,'.g', markersize=10, label='latent')
        plt.plot(self.time, self.bias +  self.post_mean, color='blue', ls = 'dotted', label='posterior mean')
        if flag == 'observed':
            error_bars = 2 * np.sqrt(np.diag(self.post_cov) + self.sigma_n**2 )  
        elif flag == 'latent':
            error_bars = 2 * np.sqrt(np.diag(self.post_cov))
        #print(f'average error_bars are: {np.mean(error_bars[0:500])}')
        plt.fill_between(self.time, self.bias + self.post_mean - error_bars, self.bias + self.post_mean + error_bars, color='blue',alpha=0.3, label='95% error bars')
        if x_true is not None:  
            plt.plot(x_true, self.bias + y_true, color='k', label=label_true)

        #plt.title(f'Observations, latent values and GP (NLL: {self.neg_log_likelihood()})')
        #plt.xlabel(self.time_label)
        #plt.ylabel(self.signal_label)
        plt.legend()
        plt.xlim([min(self.time),max(self.time)])
        plt.tight_layout()


    def print_params(self):
        if self.kernel == 'qSM':
            print(f'marginal std is: {np.sqrt(self.power)}')
            print(f'noise std is: {self.sigma_n}')
            print(' ')
            for qq in range(self.order):
                print(f'Component {qq}:')
                print(f'magnitude is: {self.theta[3*qq+0]}')
                print(f'frequency is: {self.theta[3*qq+1]}')
                print(f'lengthscale is: {self.theta[3*qq+2]}')
                print(' ')





    def set_labels(self, time_label, signal_label):
        self.time_label = time_label
        self.signal_label = signal_label


