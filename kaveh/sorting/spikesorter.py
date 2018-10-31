"""
Copyright (c) 2018 Laboratory for Computational Motor Control, Johns Hopkins School of Medicine

Author: Kaveh Karbasi <kkarbasi@berkeley.edu>
"""

import numpy as np
from sklearn.mixture import GaussianMixture
import scipy.signal
import scipy.fftpack
from scipy.stats import norm
import time

class SimpleSpikeSorter:
    """ Class that detects and sorts simple spikes"""
    def __init__(self, voltage, dt):
        """
        Object constructor
        """
        self.voltage = np.squeeze(np.array(voltage))
        self.dt = dt
        self.low_pass_filter_cutoff = 10000 #Hz
        self.high_pass_filter_cutoff = 1000 #Hz
        self.filter_order = 2
        self.num_gmm_components = 6
        self.gmm_cov_type = 'tied'
        self.pre_window = 0.0005 #s
        self.post_window = 0.005 #s
        self.minibatch_thresh = 50 #s - for spike detection: if signal length more than this, switch to minibatch GMM
        # Complex spike detection parameters:
        self.freq_range = (0, 5000) #Hz
        self.cs_num_gmm_components = 2
        self.cs_cov_type = 'tied'
        self.post_cs_pause_time = 0.015 #s

    def run(self):
	start = time.time()
        self._pre_process()
	print('Pre-process time = {}'.format(time.time() - start))
        delta = int(self.minibatch_thresh / self.dt)
        if delta >= self.voltage_filtered.size:
            self._detect_spikes()
        else:
            self._detect_spikes_minibatch()
 	print('Spike detection time = {}'.format(time.time() - start))
        self._align_spikes()
 	print('Align spikes time = {}'.format(time.time() - start))
        self._cluster_spike_waveforms_by_freq()
 	print('CS spike detection time = {}'.format(time.time() - start))
        self._cs_post_process()
 	print('CS post process time = {}'.format(time.time() - start))


    def _pre_process(self):
        """
        Pre-processing on the input voltage signal:
        Apply zero-phase linear filter
        """
        self.voltage_filtered = np.copy(self.voltage)
        [b, a] = scipy.signal.filter_design.butter(self.filter_order,
                        [2 * self.dt * self.high_pass_filter_cutoff, 2 * self.dt * self.low_pass_filter_cutoff],
                                               btype='bandpass')
        self.voltage_filtered = scipy.signal.lfilter(b, a, self.voltage_filtered)  # Filter forwards
        self.voltage_filtered = np.flipud(self.voltage_filtered)
        self.voltage_filtered = scipy.signal.lfilter(b, a, self.voltage_filtered)  # Filter reverse
        self.voltage_filtered = np.flipud(self.voltage_filtered)
        self.voltage_filtered = scipy.signal.savgol_filter(self.voltage_filtered, 5, 2, 1, self.dt)

    def _detect_spikes(self):
        """
        Preliminary spike detection using a Gaussian Mixture Model
        """
        gmm = GaussianMixture(self.num_gmm_components,
                covariance_type = 'tied').fit(self.voltage_filtered.reshape(-1,1))
        cluster_labels = gmm.predict(self.voltage_filtered.reshape(-1,1))
        cluster_labels = cluster_labels.reshape(self.voltage_filtered.shape)
        spikes_cluster = np.argmax(gmm.means_)
        all_spike_indices = np.squeeze(np.where(cluster_labels == spikes_cluster))
        # Find peaks of each spike
        peak_times,_ = scipy.signal.find_peaks(self.voltage_filtered[all_spike_indices])
        self.spike_indices = all_spike_indices[peak_times]

    # TODO
    def _detect_spikes_from_range(self, prange):
        """
        Preliminary spike detection using a Gaussian Mixture Model, using only a range of signal
        """
        voltage_signal = self.voltage_filtered[prange]
        gmm = GaussianMixture(self.num_gmm_components,
                covariance_type = 'tied').fit(voltage_signal.reshape(-1,1))
        cluster_labels = gmm.predict(voltage_signal.reshape(-1,1))
        cluster_labels = cluster_labels.reshape(voltage_signal.shape)
        spikes_cluster = np.argmax(gmm.means_)
        all_spike_indices = np.squeeze(np.where(cluster_labels == spikes_cluster))
        # Find peaks of each spike
        peak_times,_ = scipy.signal.find_peaks(voltage_signal[all_spike_indices])
        spike_indices = all_spike_indices[peak_times]
        return spike_indices

    def _detect_spikes_minibatch(self):
        """
        Loops through the entire voltage signal and detects spikes
        The loop is on each slices of the signal with size minibatch_thresh
        """
        print('Using minibatch spike detection, batch size = {}s'.format(self.minibatch_thresh))
        delta = int(self.minibatch_thresh/self.dt)
        self.spike_indices = np.array([], dtype='int64')
        for i in np.arange(0, self.voltage_filtered.size - int(10/self.dt), delta):
            curr_indices = self._detect_spikes_from_range(slice(i, i + delta)) 
            curr_indices = curr_indices + i
            self.spike_indices = np.concatenate((self.spike_indices, curr_indices), axis=None)
	
    
    def _remove_overlapping_spike_windows(self):
        """
        Removes the spike indices that have overlapping alignment windows
        """
        to_delete = []

        pre_index = int(np.round(self.pre_window / self.dt))
        post_index = int(np.round(self.post_window / self.dt))

        for i, spike_index in enumerate(self.spike_indices[1:]):
            if (spike_index - pre_index) <=(self.spike_indices[i] + post_index) or self.spike_indices[i] - pre_index < 0:
                to_delete = to_delete + [i]
        if self.spike_indices[-1] + post_index >= self.voltage.size:
            to_delete = to_delete + [self.spike_indices.size - 1]
#         if self.spike_indices[0] - pre_index < 0:
#             to_delete = [0] + to_delete
        mask = np.ones(self.spike_indices.shape, dtype=bool)
        mask[to_delete] = False
        no_overlap_spike_indices = self.spike_indices[mask]

        return no_overlap_spike_indices


    def _align_spikes(self, use_filtered = False, to_exclude = []):
        """
        Aligns the spike waveforms (from pre_window to post_windows, aligned to peak times)
        Pass to_exclude if you want to exclude from adding to the align matrix
        """
        pre_index = int(np.round(self.pre_window / self.dt))
        post_index = int(np.round(self.post_window / self.dt))
        spike_indices = self._remove_overlapping_spike_windows()
        signal_size_f = self.voltage_filtered.size
        signal_size = self.voltage.size

        if use_filtered:
            self.aligned_spikes = np.array([self.voltage_filtered[i - pre_index : i + post_index ] 
                for i in spike_indices if i not in to_exclude if (i + post_index) < signal_size_f
                    if (i - pre_index) >= 0])
        else:
            self.aligned_spikes = np.array([self.voltage[i - pre_index : i + post_index ] 
                for i in spike_indices if i not in to_exclude if (i + post_index) < signal_size
                    if (i - pre_index) >= 0])

            

    # TODO
    def _choose_num_features(self, captured_variance=0.75):
        """
        Use the number of components that captures at least captured_variance of the spike waveforms
        """
        return 0
    
    def _find_max_powers(self):
        """
        Finds and returns the maximum power of all aligned spike waveforms in a specified frequency range.
        freq_range: a tuple of frequency range boundaries (in Hz)
        """
        powers = [] 
        max_powers = []
        for wf in self.aligned_spikes:
            yf = scipy.fftpack.fft(wf)
            N = wf.size
            xf = np.linspace(0.0, 1.0 / (2.0 * self.dt), N/2)
            mask = (xf < self.freq_range[1]) & (xf >= self.freq_range[0])
            power_spectrum = 2.0/N * np.abs(yf[:N//2])
            max_powers = max_powers + [np.max(power_spectrum[mask])]
            powers.append(power_spectrum[mask])
        max_powers = np.asarray(max_powers)
        powers = np.array(powers)
        return max_powers, powers, xf[mask] 

    def _cluster_spike_waveforms_by_freq(self, plot_hist = False):
        """
        Clusters the found spikes into simple and complex, using a gmm_nc component GMM
        It uses the maximum power in the lower region of the frequency spectrum of the 
        spike waveforms
        """
        max_powers = self._find_max_powers()[0]
        gmm = GaussianMixture(self.cs_num_gmm_components, covariance_type = self.cs_cov_type).fit(max_powers.reshape(-1,1))
        cluster_labels = gmm.predict(max_powers.reshape(-1,1))
        cluster_labels = cluster_labels.reshape(max_powers.shape)

        cs_indices = self.get_spike_indices()[cluster_labels == np.argmax(gmm.means_)]
        if plot_hist:
            plt.figure()
            # uniq = np.unique(ss.d_voltage[prang] , return_counts=True)
            x = np.arange(np.min(max_powers), np.max(max_powers), 1)
            if self.cs_cov_type == 'tied':
                gauss_mixt = np.array([p * norm.pdf(x, mu, np.sqrt(gmm.covariances_.flatten())) 
                    for mu, p in zip(gmm.means_.flatten(), gmm.weights_)])
            else:
                gauss_mixt = np.array([p * norm.pdf(x, mu, sd) 
                    for mu, sd, p in zip(gmm.means_.flatten(), np.sqrt(gmm.covariances_.flatten()), gmm.weights_)])
                        
            colors = plt.cm.jet(np.linspace(0,1,len(gauss_mixt)))

            # plot histogram overlaid by gmm gaussians
            for i, gmixt in enumerate(gauss_mixt):
                    plt.plot(x, gmixt, label = 'Gaussian '+str(i), color = colors[i])

                    plt.hist(max_powers.reshape(-1,1),bins=256,density=True, color='gray')
                    axvlines(plt.gca(), gmm.means_)
                    plt.show()
        self.cs_indices = cs_indices
    
    def _cs_post_process(self):
        """
        Post processing for complex spikes.
        pause_time: the pause time that the complex spikes should produce in the spike train.
        Any detected complex spike that produces less pause than this is ignored
        """
        # Remove detected cs that don't produce a pause in simple spikes for pause_time
        to_delete = []
	spike_indices = self.get_spike_indices()
        for i, csi in enumerate(self.cs_indices):
            if (spike_indices[-1] != csi):
                if (spike_indices[np.squeeze(np.where(spike_indices == csi)) + 1] - csi) \
                   * self.dt < self.post_cs_pause_time:
                    to_delete = to_delete + [i]
        mask = np.ones(self.cs_indices.shape, dtype = bool)
        mask[to_delete] = False
        self.cs_indices = self.cs_indices[mask]

    def recluster_complex_spikes(self, freq_range=None, gmm_nc=None, cov_type=None, plot_hist = False):
        """
        Re-run complex spike clustering with new parameters for the GMM
        """
        if freq_range is not None:
            self.freq_range = freq_range
        if gmm_nc is not None:
            self.cs_num_gmm_components = gmm_nc
        if cov_type is not None:
            self.cs_cov_type = cov_type
        self._cluster_spike_waveforms_by_freq(plot_hist = plot_hist)
        self._cs_post_process()

    def get_cs_spike_indices(self):
        """
        Returns the detected complex spike indices
        """
        return self.cs_indices


    def set_spike_window(self, pre_time, post_time):
        """
        Sets the spike window for complex spike detection
        """
        self.pre_window = pre_time
        self.post_window = post_time
        self._align_spikes()

    def get_spike_indices(self, remove_overlaps=True):
        """
        Returns the detected spike indices
        remove_overlaps: If we want to remove the overlapping spikes based on the set window
        """
        if remove_overlaps:
            return self._remove_overlapping_spike_windows()
        else:
            return self.spike_indices

    def plot_spike_waveforms_average(self, **kw):
        """
        Plots the average spike wavelets of the current dataset
        """
        spikes_avg = np.mean(self.aligned_spikes, axis = 0)
        spikes_std = np.std(self.aligned_spikes, axis = 0)/np.sqrt(self.aligned_spikes.shape[0])
        x = np.arange(0,self.aligned_spikes.shape[1])
        plt.figure()
        l = plt.plot(x, spikes_avg, **kw)
        plt.fill_between(x, spikes_avg - spikes_std, spikes_avg + spikes_std, color=l[0].get_color(), alpha=0.25)
        
    def plot_spike_peaks(self, use_filtered = False , figsize = (21, 5)):
        """
        Plots the voltage signal, overlaid by spike peak times,
        overlaid by lines indicating the window around the spike peaks
        """
        if use_filtered:
            plt.figure(figsize = figsize)
            plt.plot(self.voltage_filtered)
            plt.plot(self.spike_indices, self.voltage_filtered[self.spike_indices], '.r')
            axvlines(plt.gca(), self.spike_indices + int(np.round(self.post_window/self.dt)), color='g')
            axvlines(plt.gca(), self.spike_indices - int(np.round(self.pre_window/self.dt)), color='m')
        else:
            plt.figure(figsize = figsize)
            plt.plot(self.voltage)
            plt.plot(self.spike_indices, self.voltage[self.spike_indices], '.r')
            axvlines(plt.gca(), self.spike_indices + int(np.round(self.post_window/self.dt)), color='g')
            axvlines(plt.gca(), self.spike_indices - int(np.round(self.pre_window/self.dt)), color='m')

    def plot_voltage(self, use_filtered = False, figsize = (21, 5)):
        """
        Plots the voltage trace
        """
        plt.figure(figsize = figsize)
        if use_filtered:
            plt.plot(np.arange(0, self.voltage_filtered.size) * self.dt, self.voltage_filtered)
            plt.title('Filtered voltage vs Time(s)')
            plt.xlabel('t(s)')
        else:
            plt.plot(np.arange(0, self.voltage.size) * self.dt, self.voltage)
            plt.title('Raw voltage vs Time(s)')
            plt.xlabel('t(s)')






        
        



