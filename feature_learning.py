# Using GPU enabled NMF to learn spectral features(W) and temporal activations(H) in an unsupervised manner
import torch
from pytorch-NMF.torchnmf.nmf import NMF
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from soundscape_IR.soundscape_viewer.utility import save_parameters
from soundscape_IR.soundscape_viewer.utility import gdrive_handle
from scipy.io import loadmat
from scipy.io import savemat
from sklearn.decomposition import non_negative_factorization as NMF_cpu

class save_parameters_revised:
    def __init__(self):
        self.platform='python'
    
    def supervised_nmf(self, f, W, feature_length, basis_num,H):
        self.f=f
        self.W=W
        self.time_frame=feature_length
        self.basis_num=basis_num
        self.H=H
    
    def pcnmf(self, f, W, W_cluster, source_num, feature_length, basis_num):
        self.f=f
        self.W=W
        self.W_cluster=W_cluster
        self.k=source_num
        self.time_frame=feature_length
        self.basis_num=basis_num

    def LTS_Result(self, LTS_median, LTS_mean, f, link):
        self.LTS_median = LTS_median
        self.LTS_mean = LTS_mean
        self.f = f
        self.link = link

    def LTS_Parameters(self, FFT_size, overlap, sensitivity, sampling_freq, channel):
        self.FFT_size=FFT_size
        self.overlap=overlap 
        self.sensitivity=sensitivity 
        self.sampling_freq=sampling_freq 
        self.channel=channel


class nmf_gpu:
  def __init__(self, feature_length=1, basis_num=60):
      self.basis_num=basis_num
      self.feature_length=feature_length

  #def feature_learning(self, input_data, f, alpha=1, l1_ratio=1, max_iter=200, beta=1, kernel='gpu'):
  def feature_learning(self, input_data, f, max_iter=200, beta=2, sW=None, sH=None, kernel='gpu'):
      # Preparing data
      self.f=f
      self.time_vec=input_data[:,0:1]
      input_data=input_data[:,1:].T
      baseline=input_data.min()
      input_data=input_data-baseline
      print('Run NMF')

      # Modify the input data based the feature width
      if self.feature_length>1:
        input_data=self.matrix_conv(input_data)

      print('Feature learning...')
      if kernel=='gpu':
        # NMF-based feature learning
        data = torch.FloatTensor(input_data)
        net = NMF(data.shape, rank=self.basis_num).cuda()
        #net.fit_transform(data.cuda(), verbose=True, max_iter=max_iter, tol=1e-18, alpha=alpha, l1_ratio=l1_ratio, beta=beta)
        net.fit_transform(data.cuda(), verbose=True, max_iter=max_iter, beta=beta, sW=sW, sH=sH, sparse=True)
        self.W, self.H = net.W.detach().cpu().numpy(), net.H.detach().cpu().numpy()
        data = data.detach().cpu()
      elif kernel=='cpu':
        self.W, self.H, _ = NMF_cpu(input_data, n_components=self.basis_num, beta_loss=beta, alpha=alpha, l1_ratio=l1_ratio)

      print('Done')

  def reconstruct(self):
      output = np.dot(self.W, self.H)
      temp=np.zeros((len(self.f), self.H.shape[1]+1-self.feature_length))
      for x in range(self.feature_length):
        #temp=temp+output[x*len(self.f):(x+1)*len(self.f),x:self.H.shape[1]+1-self.feature_length+x]
        temp=temp+output[(self.feature_length-(x+1))*len(self.f):(self.feature_length-x)*len(self.f),x:self.H.shape[1]+1-self.feature_length+x]
      output=np.divide(temp,self.feature_length)
      output[np.isnan(output)]=0
      return output

  def matrix_conv(self, input_data):
      # Adding time-series information for each vector
      matrix_shape=input_data.shape
      data=np.zeros((matrix_shape[0]*self.feature_length, matrix_shape[1]-1+self.feature_length))
      for x in range(self.feature_length):
        data[(self.feature_length-(x+1))*matrix_shape[0]:(self.feature_length-x)*matrix_shape[0],x:matrix_shape[1]+x]=input_data
      return data

  def plot_nmf(self, plot_type='Both', W_list=None):
      # Plot the spectral features(W) and temporal activations(H) learned by using the NMF
      W=np.array(self.W)
      W_num=W.shape[1]
      if self.W.shape[0]>len(self.f):
        if isinstance(W_list,np.ndarray):
          W=np.array(W[:,W_list])
        W=np.vstack((np.zeros((len(self.f),W.shape[1])), W)).T.reshape(1,-1)
        W=W.reshape((-1,len(self.f))).T
      elif self.W.shape[0]==len(self.f):
        if W_list:
          W=W[:,W_list]

      if isinstance(W_list,np.ndarray):
        H=np.array(self.H[W_list,:])
      else:
        H=np.array(self.H)

      # plot the features
      if plot_type=='Both':
        fig, (ax1, ax2) = plt.subplots(nrows=2,figsize=(20, 12))
      elif plot_type=='W':
        fig, ax1 = plt.subplots(figsize=(300, 50))
      elif plot_type=='H':
        fig, ax2 = plt.subplots(figsize=(14, 6))

      if plot_type=='Both' or plot_type=='W':
        im = ax1.imshow(W, origin='lower',  aspect='auto', cmap=cm.jet,
                        extent=[0, W_num, self.f[0], self.f[-1]], interpolation='none')
        ax1.set_title('W')
        ax1.set_ylabel('Frequency')
        ax1.set_xlabel('Basis')
        cbar1 = fig.colorbar(im, ax=ax1)
        cbar1.set_label('Relative amplitude')

      if plot_type=='Both' or plot_type=='H':
        im2 = ax2.imshow(H, origin='lower',  aspect='auto', cmap=cm.jet,
                        extent=[self.time_vec[0], self.time_vec[-1], 0, W_num], interpolation='none')
        ax2.set_title('H')
        ax2.set_ylabel('Basis')
        ax2.set_xlabel('Time')
        cbar2 = fig.colorbar(im2, ax=ax2)
        cbar2.set_label('Relative amplitude')

  def save_model(self, filename='NMF_model.mat', folder_id=[]):
      from dolphin_whistle.feature_learning import save_parameters_revised
      #import save_parameters
      nmf_model=save_parameters_revised()
      nmf_model.supervised_nmf(self.f, self.W, self.feature_length, self.basis_num,self.H)
      savemat(filename, {'save_model':nmf_model})
      print('Successfully save to '+filename)

      # save the result in Gdrive as a mat file
      if folder_id:
        Gdrive=gdrive_handle(folder_id)
        Gdrive.upload(filename)

  def load_model(self, filename):
      model = loadmat(filename)
      self.f=model['save_model']['f'].item()[0]
      self.W=model['save_model']['W'].item()
      self.H=model['save_model']['H'].item()
      self.feature_length=model['save_model']['time_frame'].item()[0][0]
      self.basis_num=model['save_model']['basis_num'].item()[0][0]
