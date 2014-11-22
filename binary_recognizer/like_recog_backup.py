import sys, os, mne, re, math, scipy, pylab
import numpy as np 
import matplotlib.pyplot as plt
# from scipy.fftpack import fft, ifft
import scipy.fftpack
from numpy.fft import fft, fftshift
from scipy import signal, pi
from scipy.signal import butter, lfilter
from spectrum import *
from sklearn import svm


sys.path.insert(0, '/Users/ziqipeng/Dropbox/bci/system_zero/backend/emotiv')
# print "################# %s" % os.path.dirname(os.path.realpath(__file__))
# sys.path.append('../backend/emotiv')
import data_loader_emotiv as dle

class LikeRecog(object):

	def __init__(self):
		self.el = dle.DataLoaderEmotiv()
		self.data_dic = self.el.build_dic()
		# self.dic = self.el.edf_loader()
		self.data = []
		self.label = []
		t = 0
		# for ck, cv in self.data_dic.iteritems():
		# 	if t == 0:
		# 		self.length = len(cv)
		# 	self.data.append(cv)
		# 	self.label.append(ck)
		# 	t += 1
		# self.terms = math.floor(self.length/(128/4))
		# self.train_terms = math.floor(self.length/((60 * 2 * 128) + (12 * 128)))
		# like for 0, dislike for 2, neutral for 1
		self.tag_dic = {'video0': 2, 
						'video1': 0, 
						'video2': 0, 
						'video3': 0, 
						'video4': 1, 
						'video5': 0, 
						'video6': 1, 
						'video7': 0, 
						'video8': 1, 
						'video9': 1, 
						'video10': 2, 
						'video11': 2, 
						'video13': 0, 
						'video14': 2}

	def trim_data(self):
		for k, v in self.data_dic.iteritems():
			one_video_data = []
			one_video_ch = []
			for tag in self.tag_dic:
				if tag in k:
					tag_label = self.tag_dic[tag]
					# labe - like for 1, other for 0
					if tag_label ==  0:
						label = 1
					else:
						label = 0
			for ch, data in v.iteritems():
				one_video_ch.append(ch)
				one_video_data.append(data)
				video_terms = int(math.floor(len(data)/(128.0/2)))
				print type(data)
				print len(data)
			start = 0
			tags = [label*128]
			for i in range(int(video_terms)):
				wid_sec = []
				for one_ch_data in one_video_data: 
					wid_sec_ch = one_ch_data[start:(start + 128)]
					wid_sec.append(wid_sec_ch)




		# self.trim_data = self.data[(128*10):(128*10 + 128*60)]


	def run_train(self):
		pass


	def realtime_window(self):
		pass

	def train_set(self):
		self.tterms = self.length/(128*(60+12))

	def svm(self, data):
		llist = [1] * 128 * 60 
		llist.append([0] * 128 * 60)
		llist = llist * 5
		fdic = self.normalize_train()
		x = []
		for k, v in fdic.iteritems():
			x.append(v)
		x = np.asarray(x)
		llist = np.asarray(llist)
		clf = svm.NuSVC()
		clf.fit(x, llist)
		return clf

	def remove_artifact(self, data):
		pass

	def emg(self, data):
		pass

	def eog(self, data):
		pass

	def ann(self, data):
		pass

	def pca(self, data):
		pass

	def ica(self, data):
		pass

	def knn(self, data):
		pass

	def km(self, data):
		pass

	def hmm(self, data):
		pass

	def classify(self, data):
		pass

	def normalize_train(self):
		wid = np.asarray(self.data, dtype=np.float64)
		pmax = sys.float_info.min
		pmin = sys.float_info.max
		for i in range(3 * self.train_terms):
			for n in range(2):
				s = 0
				for i in range(60):
					wid = wid[s, s + 128]
					psd_wid = self.feature_psd(wid)
					for k, v in psd_wid.iteritems():
						if np.amax(v) > pmax:
							pmax = np.amax(v)
						if np.amin(v) < pmin:
							pmin = np.min(v)
					s += 64
					if i == 59:
						s = 0
		nm_psd = dict((el, []) for el in psd_w.keys()) 
		for i in range(3 * self.train_terms):
			for n in range(2):
				s = 0
				if n == 0:
					label = 'h_'
				else:
					label = 'u_'
				for i in range(60):
					w = wid[s, s + 128]
					psd_w = self.feature_psd(w)
					for k, v in psd_w.iteritems():
						nm_psd[label + k] = np.asarray(list(map((lambda x: (x - pmin)/(pmax - pmin)), v)))
					s += 64
					if i == 59:
						s = 0
		return nm_psd

	def normalize_dev(self, type):
		wid = np.asarray(self.data, dtype=np.float64)
		pmax = sys.float_info.min
		pmin = sys.float_info.max
		s = 0
		for i in range(self.terms):
			wid = wid[s, s + 128]
			psd_wid = self.feature_psd(wid)
			for k, v in psd_wid.iteritems():
				if np.amax(v) > pmax:
					pmax = np.amax(v)
				if np.amin(v) < pmin:
					pmin = np.min(v)
			s += 64
		nm_psd = dict((el, []) for el in psd_w.keys()) 
		s = 0
		for i in range(self.terms):
			w = wid[s, s + 128]
			psd_w = self.feature_psd(w)
			for k, v in psd_w.iteritems():
				nm_psd[k] = np.asarray(list(map((lambda x: (x - pmin)/(pmax - pmin)), v)))
			s += 64
		return nm_psd

	def feature_nmf(self, wid):
		pass

	def feature_psd(self, wid, ch_name):
		freq = np.fft.fftfreq(128, 1.0/128)
		fftw = []
		for i in wid:
			ffti = fft(i)
			fftw.append(ffti)
		fftwid = np.asarray(fftw, dtype=np.float64)
		fdic = {}
		ch0 = 0
		for cn, d in zip(ch_name, wid):
			k0 = cn + "_delta"
			delta = mne.filter.band_pass_filter(x=d, Fs=128, Fp1=0.5, Fp2=4, l_trans_bandwidth=0.0)
			fdic[k0] = delta
			fft_delta = fft(delta)
			freq = np.fft.fftfreq(delta.shape[-1])
			# plt.figure(1)
			# if ch0 == 0:
			# 	plt.subplot(211)
			# 	plt.plot(np.arange(128), delta)
			# 	plt.subplot(212)
			# 	plt.plot(freq, fft_delta)
			# plt.show()
			# ch0 += 1
			k1 = cn + "_theta"
			theta = mne.filter.band_pass_filter(x=d, Fs=128, Fp1=4, Fp2=8)
			fdic[k1] = theta
			k2 = cn + "_alpha"
			alpha = mne.filter.band_pass_filter(x=d, Fs=128, Fp1=8, Fp2=16)
			fdic[k2] = alpha
			k3 = cn + "_beta"
			beta = mne.filter.band_pass_filter(x=d, Fs=128, Fp1=16, Fp2=32)
			fdic[k3] = beta
			k4 = cn + "_gamma"
			gamma = mne.filter.band_pass_filter(x=d, Fs=128, Fp1=32, Fp2=63.5)
			fdic[k4] = gamma
		psd_dic = dict((el, []) for el in fdic.keys())
		check = 0
		for k, v in fdic.iteritems():
			f, psd_dic[k] = signal.welch(v, fs=128, nperseg=128)
			# if check==0:
				# plt.figure(212)
				# plt.semilogy(f, psd_dic[k])
				# plt.xlabel('frequency [Hz]')
				# plt.ylabel('PSD')
			print type(psd_dic[k])
			print len(psd_dic[k])
				# print psd_dic[k]
			check += 1
		# print psd_dic
		# print check
		return psd_dic

	def fft_train(self):
		tw = []
		for i in range(0, self.terms):
			pass 

	def spectrum_test(self):
		data = data_cosine(N=1024, A=0.1, sampling=1024, freq=200)
		print data
		print len(data)
		print type(data)
		p = Periodogram(data, sampling=1024)
		p()
		p.plot(marker='o')

	def spectrum_test1(self):
		ar, ma, rho = arma_estimate(marple_data, 15, 15, 30)

	def psd(self):
		el = dle.DataLoaderEmotiv() #DataLoaderEmotiv()
		dic = {}
		dic = el.edf_loader()

		for k in dic:
			if k == "F3":
				x = np.fft(dic[k])
				plt.plot(abs(x))
				pl.show()

if __name__ == "__main__":
	lr = LikeRecog()
	lr.trim_data()
	# lr.feature_psd(lr.data)
	# lr.svm("")
	# lr.fft()

