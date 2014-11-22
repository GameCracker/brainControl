import sys, os, mne, re, math, scipy, pylab, inspect
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib as mpl
# from scipy.fftpack import fft, ifft
import scipy.fftpack
from numpy.fft import fft, fftshift
from scipy import signal, pi
from scipy.signal import butter, lfilter
from spectrum import *
from sklearn import svm
from mpl_toolkits.mplot3d import proj3d


sys.path.insert(0, '/Users/ziqipeng/Dropbox/bci/startupweekend/backend/emotiv')
# print "################# %s" % os.path.dirname(os.path.realpath(__file__))
# sys.path.append('../backend/emotiv')
import data_loader_emotiv as dle

class LikeRecog(object):

	def __init__(self):
		self.el = dle.DataLoaderEmotiv()
		self.data_dic = self.el.build_dic()
		# self.rt_dic = self.el.realtime_data()
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
		self.feature_list = ['F3_theta', 'T8_gamma', 'F7_gamma', 'FC5_alpha', 'FC5_delta', 'F3_delta', 'T8_theta', 'O2_theta', 'AF3_alpha', 'O1_theta', 'AF3_beta', 'AF3_gamma', 'T8_beta', 'F8_theta', 'F7_beta', 'FC6_gamma', 'F3_beta', 'F7_theta', 'F7_delta', 'O2_beta', 'AF4_delta', 'T8_alpha', 'F4_delta', 'P8_delta', 'O1_alpha', 'P8_gamma', 'FC6_delta', 'O2_delta', 'F8_beta', 'P8_beta', 'T7_delta', 'P7_alpha', 'T7_theta', 'P7_gamma', 'AF4_beta', 'P8_theta', 'F7_alpha', 'O1_beta', 'F3_gamma', 'FC5_theta', 'F4_theta', 'AF4_theta', 'P7_delta', 'FC6_beta', 'T7_gamma', 'F4_beta', 'AF4_alpha', 'F4_gamma', 'O1_gamma', 'AF3_delta', 'FC6_alpha', 'F8_gamma', 'O2_alpha', 'FC6_theta', 'T8_delta', 'F8_delta', 'P7_theta', 'F4_alpha', 'O2_gamma', 'F8_alpha', 'F3_alpha', 'P8_alpha', 'P7_beta', 'AF3_theta', 'O1_delta', 'FC5_beta', 'AF4_gamma', 'T7_beta', 'T7_alpha', 'FC5_gamma']
		self.feature_idx = range(70)
		self.feature_dict = dict(zip(self.feature_list, self.feature_idx))
		# self.feature_idx = 

	def predict(self):
		# self.pre_data = dle.DataLoaderEmotiv()
		# self.pre_data_dic = self.pre_data
		pass

	def split_secwid_train(self):
		print "Load video based EEG data and feature selection..."
		self.all_tags = []
		self.all_features = []
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
				video_terms = int(math.floor(len(data)/64))
				# print type(data)
				# print len(data)
			start = 0
			# tags = [label]*128
			# print "video_term: %d" % video_terms
			for i in range(video_terms):
				wid_sec = []
				for one_ch_data in one_video_data: 
					wid_sec_ch = one_ch_data[start:(start + 128)]
					wid_sec.append(wid_sec_ch)
				# print "i = %d" % i
				start += 64
				# print np.shape(wid_sec)
				# print len(tags)
				if np.shape(wid_sec)[1] == 128:
					# print "hiiiiiiiiiiii"
					# norm_sec_dict = self.normalize_train(wid_sec, one_video_ch, video_terms)
					sec_psd_dic = self.feature_psd(wid_sec, one_video_ch)
					sec_psd_list = []
					# print "&&&&&&&&&&&&&&&&&&&&&&&"
					# print sec_psd_dic.keys()
					# print len(sec_psd_dic.keys())
					# print "&&&&&&&&&&&&&&&&&&&&&&&"
					new_sec_dic = {}
					for k, v in sec_psd_dic.iteritems():
						new_k = self.feature_dict[k]
						new_sec_dic[new_k] = v
					key_list = new_sec_dic.keys()
					key_list.sort()
					# if count == 0:
					for key in key_list:
						# all_features.append(new_sec_dic[key])
						sec_psd_list.append(new_sec_dic[key])
					print "***********************************"
					print np.shape(sec_psd_list)
					print "***********************************"
					self.all_features.extend(sec_psd_list)
					self.test_data = self.all_features[0]
					# self.tl = 1
					self.test_data1 = self.all_features[1]
					tags = [label]*70
					self.all_tags.extend(tags)
					# dic_ir = iter(sorted(sec_psd_dic.iteritems()))
					# print "++++++++++++++++++++++++++++++++++"
					# print len(dic_ir.next())
					# print dic_ir.next()[0]
					# print "++++++++++++++++++++++++++++++++++"
		# self.trim_data = self.data[(128*10):(128*10 + 128*60)]

	def run_train(self):
		self.split_secwid_train()
		print "Training classifier..."
		clf = self.svm()

	def realtime_window(self):
		sys.path.insert(0, '/Users/ziqipeng/Dropbox/bci/startupweekend/backend/emotiv')
		self.rt_dic = self.el.realtime_data()
		for k, v in self.rt_dic.iteritems():
			pass

	def train_set(self):
		self.tterms = self.length/(128*(60+12))

	def svm(self):
		# llist = [1] * 128 * 60 
		# llist.append([0] * 128 * 60)
		# llist = llist * 5
		# fdic = self.normalize_train()
		# x = []
		# for k, v in fdic.iteritems():
		# 	x.append(v)
		# print "++++++++++++++++++++++++++++++++++"
		# print np.shape(self.all_tags)
		# print np.shape(self.all_features)
		# print "++++++++++++++++++++++++++++++++++"
		xs = np.asarray(self.all_features)
		ys = np.asarray(self.all_tags)
		clf = svm.NuSVC()
		clf.fit(xs, ys)
		print "Done with classifier!"
		print "Starting predict on test feature 0: "
		result = clf.predict(self.test_data)
		print result
		if result == 0:
			print "predicted as dislike!"
		else:
			print "predicted as like!"
		print "predict on test feature 1: "
		result1 = clf.predict(self.test_data1)
		print result1
		if result == 0:
			print "predicted as dislike!"
		else:
			print "predicted as like!"
		return clf

	def remove_artifact(self, data):
		pass

	def emg(self, data):
		pass

	def eog(self, data):
		pass

	def ann(self, data):
		pass

	def pca(self):
		pass

	def pca_util(self, d):
		np.random.seed(4294967295)
		mu_vec1 = np.array([0, 0, 0])
		cov_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
		class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 20).T
		assert class1_sample.shape == (3, 20), "The matrix has not the dimension 3x20"
		mu_vec2 = np.array([1, 1, 1])
		cov_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
		class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 20).T
		# print class2_sample
		assert class1_sample.shape == (3,20), "The matrix has not the dimensions 3x20"
		from mpl_toolkits.mplot3d import Axes3D
		from mpl_toolkits.mplot3d import proj3d
		fig = plt.figure(figsize=(8, 8))
		ax = fig.add_subplot(111, projection='3d')
		mpl.rcParams['legend.fontsize'] = 10
		ax.plot(class2_sample[0, :], class2_sample[1, :], class1_sample[2, :], 'o', markersize=8, color='blue', alpha=0.5, label='class1')
		ax.plot(class2_sample[0, :], class2_sample[1, :], class2_sample[2, :], '^', markersize=8, color='red', alpha=0.5, label='class2')
		plt.title('samples for class 1 and 2')
		ax.legend(loc='upper right')
		# plt.draw()
		# plt.show()
		all_samples = np.concatenate((class1_sample, class2_sample), axis=1)
		print "all_samples"
		print all_samples
		assert all_samples.shape == (3, 40), "The matrix has not the 3x40"
		self.pca_dev(all_samples, d)

	def line(self):
		inspect.currentframe().f_back.f_lineno

	def pca_dev(self, all_samples, d):
		print "compare"
		mean_list = []
		for layer in all_samples:
			mean_list.append([np.mean(layer)])
			# print layer
			# break
		mean_vector = np.array(mean_list)
		# print "mean_ary"
		# print mean_ary
		# print all_samples[0, :]
		# mean_x = np.mean(all_samples[0, :])
		# mean_y = np.mean(all_samples[1, :])
		# mean_z = np.mean(all_samples[2, :])
		# mean_vector = np.array([[mean_x], [mean_y], [mean_z]])
		# print "mean_vector"
		# print mean_vector
		# print('mean vec: \n', mean_vector)
		# print all_samples.shape
		n = len(mean_vector)
		scatter_matrix = np.zeros((n, n))
		# print "scatter_matrix 1st"
		# print scatter_matrix
		# print all_samples.shape[1]
		for i in range(all_samples.shape[1]):
			# if i == 1:
			# 	print all_samples[:, i]
			n = len(all_samples[:, i])
			scatter_matrix += (all_samples[:, i].reshape(n, 1) - mean_vector).dot((all_samples[:, i].reshape(n, 1) - mean_vector).T)
		# print 'Scatter matrix: \n', scatter_matrix
		# alternative
		# print "cov_mat"
		# print all_samples[0, :]
		# cov_mat = np.cov([all_samples[0, :], all_samples[1, :], all_samples[2, :]])
		# print 'Covariance Matrix:\n', cov_mat
		eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)
		# eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
		# print self.line()
		# for i in range(len(eig_val_sc)):
			# n = len(eig_vec_sc[:, i])
			# print n
			# eigvec_sc = eig_vec_sc[:, i].reshape(1, n).T
			# eigvec_cov = eig_vec_cov[:, i].reshape(1, 3).T
			# assert eigvec_sc.all() == eigvec_cov.all(), 'eigenvectors are not identifcal'
		for i in range(len(eig_val_sc)):
			n = len(eig_vec_sc[:, i])
			eigv = eig_vec_sc[:, i].reshape(1, n).T
			np.testing.assert_array_almost_equal(scatter_matrix.dot(eigv), eig_val_sc[i]*eigv, decimal=6, err_msg='', verbose=True)
		from matplotlib.patches import FancyArrowPatch


		class Arrow3D(FancyArrowPatch):

			def __init__(self, xs, ys, zs, *args, **kwargs):
				FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
				self._verts3d = xs, ys, zs

			def draw(self, renderer):
				xs3d, ys3d, zs3d = self._verts3d
				xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
				self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
				# FancyArrowPatch.draw(self, renderer)

		fig = plt.figure(figsize=(7, 7))
		ax = fig.add_subplot(111, projection='3d')
		ax.plot(all_samples[0, :], all_samples[1, :], all_samples[2, :], 'o', markersize=8, color='green', alpha=0.2)
		# ax.plot([mean_x], [mean_y], [mean_z], 'o', markersize=10, color='red', alpha=0.5)
		# for v in eig_vec_sc.T:
		# 	a = Arrow3D([mean_x, v[0]], [mean_y, v[1]], [mean_z, v[2]], mutation_scale=20, lw=3, color="r")
		# 	ax.add_artist(a)
		# ax.set_xlabel('x_values')
		# ax.set_ylabel('y_values')
		# ax.set_zlabel('z_values')
		# plt.title('Eigenvectors')
		# plt.show()
		for ev in eig_vec_sc:
			np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))
		eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:, i]) for i in range(len(eig_val_sc))]
		eig_pairs.sort()
		eig_pairs.reverse()
		for i in eig_pairs:
			print(i[0])
		print eig_pairs
		# print eig_pairs[0][1].reshape(3, 1)
		n = len(eig_pairs[0][1])
		print 'eig_pairs\n', eig_pairs
		# print d
		for i in range(d - 1):
			if i == 0:
				matrix_w = np.hstack((eig_pairs[i][1].reshape(n, 1), eig_pairs[i + 1][1].reshape(n, 1)))
				# print "i=%d" % i
			else:
				matrix_w = np.hstack((matrix_w, eig_pairs[i + 1][1].reshape(n, 1)))
				# print 'i=%d' % i
		transformed = matrix_w.T.dot(all_samples)
		print 'matrix_w.shape\n', np.shape(matrix_w)
		print 'all_samples.shape\n', np.shape(all_samples)
		assert transformed.shape == (d, (np.shape(all_samples))[1])
		plt.plot(transformed[0, 0:20], transformed[1, 0:20], 'o', markersize=7, color='orange', alpha=0.5, label='class1')
		plt.plot(transformed[0, 20:40], transformed[1, 20:40], '^', markersize=7, color='purple', alpha=0.5, label='class2')
		plt.xlim([-4, 4])
		plt.ylim([-4, 4])
		plt.xlabel('x_values')
		plt.ylabel('y_values')
		plt.legend()
		plt.title('Transformed samples with class labels')
		plt.draw()
		plt.show()
		# print matrix_w

	def pca_built(self, all_samples):
		from sklearn.decomposition import PCA as sklearnPCA
		sklearn_pca = sklearnPCA(n_components=2)
		sklearn_transf = sklearn_pca.fit_transform(all_samples.T)
		sklearn_transf = sklearn_transf*(-1)
		plt.plot(sklearn_transf[0:20, 0], sklearn_transf[0:20, 1], 'o', markersize=7, color='yellow', alpha=0.5, label='class1')
		plt.plot(sklearn_transf[20:40, 0], sklearn_transf[20:40, 1], '^', markersize=7, color='black', alpha=0.5, label='class2')
		plt.xlabel('x_values')
		plt.ylabel('y_values')
		plt.xlim([-4, 4])
		plt.ylim([-4, 4])
		plt.legend()
		plt.title('Transformed samples with class labels from built PCA')
		plt.draw()
		plt.show()

	def ica(self, data):
		pass

	def knn(self, data):
		pass

	def km(self, data):
		pass

	def hmm(self, data):
		pass

	def nmf(self, data):
		pass

	def classify(self, data):
		pass

	def normalize_train(self, wid, ch_name, terms):
		wid = np.asarray(self.data, dtype=np.float64)
		pmax = sys.float_info.min
		pmin = sys.float_info.max
		for i in range(7 * terms):
			for n in range(2):
				s = 0
				for i in range(60):
					wid = wid[s, s + 128]
					psd_wid = self.feature_psd(wid, ch_name)
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
					psd_w = self.feature_psd(w, ch_name)
					for k, v in psd_w.iteritems():
						nm_psd[label + k] = np.asarray(list(map((lambda x: (x - pmin)/(pmax - pmin)), v)))
					s += 64
					if i == 59:
						s = 0
		# return a normalized dict
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

	def alpha_detection(self, data):

	def beta_detection(self, data):

	def feature_psd(self, wid, ch_name):
		# print "hiiiiiiiiiiii"
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
			# print type(psd_dic[k])
			# print len(psd_dic[k])
				# print psd_dic[k]
			check += 1
		# print psd_dic
		# for k, v in psd_dic.iteritems():
		# print "@@@@@@@@@@@@@@@@@@@"
		# print psd_dic.keys()
		# print "@@@@@@@@@@@@@@@@@@@"
		# print check
		# print "yesyesyes"
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
	lr.run_train()
	# lr.pca_util(2)
	# lr.feature_psd(lr.data)
	# lr.svm("")
	# lr.fft()

