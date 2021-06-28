
"""
Soundscape information retrieval
Author: Tzu-Hao Harry Lin (schonkopf@gmail.com)
"""

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import loadmat
from scipy.io import savemat
from scipy.fftpack import fft
from sklearn.decomposition import non_negative_factorization as NMF
from sklearn.decomposition.nmf import _initialize_nmf as NMF_init
from sklearn.decomposition.nmf import _update_coordinate_descent as basis_update
from sklearn.utils import check_array
from soundscape_IR.soundscape_viewer.utility import save_parameters
from soundscape_IR.soundscape_viewer.utility import gdrive_handle
import datetime


    
class supervised_nmf:
  def __init__(self, feature_length=1, basis_num=60):
    self.basis_num=basis_num
    self.feature_length=feature_length
  
  def reconstruct(self, source=None):
    if source:
      output = np.dot(self.W[:,self.W_cluster==source-1], self.H[self.W_cluster==source-1,:])
    else:
      output = np.dot(self.W, self.H)
    temp=np.zeros((len(self.f), self.H.shape[1]+1-self.feature_length))
    for x in range(self.feature_length):
      temp=temp+output[(self.feature_length-(x+1))*len(self.f):(self.feature_length-x)*len(self.f),x:self.H.shape[1]+1-self.feature_length+x]
    output=np.divide(temp,self.feature_length)
    output[np.isnan(output)]=0
    return output

  def nmf_output(self, data, time_vec, baseline=0):
    self.original_level = 10*np.log10((10**(data.T[:,1:]/10)).sum(axis=1))
    separation=np.zeros(self.source_num, dtype=np.object)
    relative_level=np.zeros(self.source_num, dtype=np.object)
    matrix_shape=data.shape

    # Use a ratio mask (e.g., S1 = V*((W1*H1)/(W*H))) to separate different sources
    source0 = self.reconstruct() #np.dot(self.W, self.H)
    for run in range(self.source_num):
      source = self.reconstruct(source=run+1) #np.dot(self.W[:,self.W_cluster==run],self.H[self.W_cluster==run,:])
      mask=np.divide(source,source0)
      mask[np.isnan(mask)]=0
      separation[run] = np.hstack((np.reshape(time_vec,(-1,1)), np.multiply(data,mask).T+baseline))
      relative_level[run] = 10*np.log10((10**(separation[run][:,1:]/10)).sum(axis=1))
    self.separation=separation
    self.relative_level=relative_level

  def plot_nmf(self, plot_type='W', source=None, time_range=[], fig_width=14, fig_height=6):      
    # Choose source accoridng to W_cluster
    if source:
      W_list=np.where(self.W_cluster==source-1)[0]
    else:
      W_list=np.arange(self.W.shape[1])

    # Only display part of the result
    H_list=np.arange(len(self.time_vec))
    if time_range:
      H_list=np.where((self.time_vec>=time_range[0])*(self.time_vec<time_range[1])==1)[0]

    # Prepare W
    if self.W.shape[0]>len(self.f):
      W=np.vstack((np.zeros((len(self.f),len(W_list))), self.W[:,W_list])).T.reshape(1,-1)
      W=W.reshape((-1,len(self.f))).T
    elif self.W.shape[0]==len(self.f):
      W=np.array(self.W[:,W_list])
        
    # Plot
    x_lim=np.array([self.time_vec[H_list[0]], self.time_vec[H_list[-1]]])
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    if plot_type=='W':
      im = ax.imshow(W, origin='lower',  aspect='auto', cmap=cm.jet,
                      extent=[0, len(W_list), self.f[0], self.f[-1]], interpolation='none')
      ax.set_ylabel('Frequency')
      ax.set_xlabel('Basis')
      cbar = fig.colorbar(im, ax=ax)
      cbar.set_label('Amplitude')

    elif plot_type=='H':
      im = ax.imshow(self.H[W_list,:][:,H_list+int(self.feature_length/2)], origin='lower',  aspect='auto', cmap=cm.jet,
                       extent=[x_lim[0], x_lim[1], 0, len(W_list)], interpolation='none')
      ax.set_ylabel('Basis')
      ax.set_xlabel('Time')
      cbar = fig.colorbar(im, ax=ax)
      cbar.set_label('Amplitude')
      
    elif plot_type=='reconstruction':
      im = ax.imshow(self.reconstruct(source=source)[:,H_list], origin='lower',  aspect='auto', cmap=cm.jet,
                      extent=[x_lim[0], x_lim[1], self.f[0], self.f[-1]], interpolation='none')
      ax.set_ylabel('Frequency')
      ax.set_xlabel('Time')
      cbar = fig.colorbar(im, ax=ax)
      cbar.set_label('Amplitude')

    elif plot_type=='separation':
      im = ax.imshow(self.separation[source-1][H_list,:][:,1:].T, origin='lower',  aspect='auto', cmap=cm.jet,
                      extent=[x_lim[0], x_lim[1], self.f[0], self.f[-1]], interpolation='none')
      ax.set_ylabel('Frequency')
      ax.set_xlabel('Time')
      cbar = fig.colorbar(im, ax=ax)
      cbar.set_label('Amplitude')

  def learn_feature(self, input_data, f, alpha=1, l1_ratio=1, beta=2, method='NMF', show_result=True):   
    self.f=f
    self.method=method
    self.time_vec=input_data[:,0:1]
    
    if method=='NMF':
      input_data=input_data[:,1:].T
      baseline=input_data.min()
      input_data=input_data-baseline

      # Modify the input data based the feature width
      if self.feature_length>1:
        input_data=pcnmf(feature_length=self.feature_length).matrix_conv(input_data)
      
      # NMF-based feature learning
      print('Feature learning...')
      self.W, self.H, _ = NMF(input_data, n_components=self.basis_num, beta_loss=beta, alpha=alpha, l1_ratio=l1_ratio)
      self.source_num = 1
      self.W_cluster=np.zeros(self.basis_num)
      print('Done')
      if show_result:
        # Plot the spectral features(W) and temporal activations(H) learned by using the NMF
        if self.W.shape[0]>len(f):
          W=np.vstack((np.zeros((len(f),self.W.shape[1])), self.W)).T.reshape(1,-1)
          W=W.reshape((-1,len(f))).T
        elif self.W.shape[0]==len(f):
          W=self.W
        
        # plot the features
        self.plot_nmf(plot_type='W')
        self.plot_nmf(plot_type='reconstruction')

    elif method=='PCNMF':
      pcnmf_model=pcnmf(feature_length=self.feature_length, basis_num=self.basis_num, alpha=alpha, beta_loss=beta, sparseness=l1_ratio)
      self.W, self.H, self.W_cluster = pcnmf_model.unsupervised_separation(input_data, f, source_num=2)
      self.source_num = 2
      if show_result:
        pcnmf_model.plot_pcnmf(source=1)
        pcnmf_model.plot_pcnmf(source=2)

  def specify_target(self, index):
    if self.method=='NMF':
      print("Only 1 target source")
      self.W_cluster=np.ones(self.basis_num)*0
    elif self.method=='PCNMF':
      print("Among the 2 sources, source #" +str(index) + " is the target source.")
      if index == 1:
        self.W_cluster=np.abs(self.W_cluster-1)
  
  def merge(self, model):
    current_source=np.max(self.W_cluster)
    for i in range(0, len(model)):
      self.W = np.hstack((self.W, model[i].W))
      if model[i].method=='NMF':
        self.W_cluster=np.hstack((self.W_cluster, model[i].W_cluster+i+1+current_source))
      elif model[i].method=='PCNMF':
        temp=np.array(model[i].W_cluster)+i+current_source
        temp[model[i].W_cluster==0]=0
        self.W_cluster=np.hstack((self.W_cluster, temp))
      
    self.source_num = int(np.max(self.W_cluster)+1)
    self.basis_num = len(self.W_cluster)

  def supervised_separation(self, input_data, f, iter=50, adaptive_alpha=0, additional_basis=0):
    self.f=f    
    self.time_vec=input_data[:,0:1]
    self.adaptive_alpha=adaptive_alpha
    self.additional_basis=additional_basis
    input_data=input_data[:,1:].T
    baseline=input_data.min()
    input_data=input_data-baseline
          
    # Modify the input data based on the feature length
    data=pcnmf(feature_length=self.feature_length).matrix_conv(input_data)
    
    # Check the learning rate (adaptive_alpha) for each source (adaptive NMF)
    if isinstance(adaptive_alpha, int) or isinstance(adaptive_alpha, float):
      adaptive_alpha = np.ones(self.source_num)*adaptive_alpha
    else:
      if len(adaptive_alpha) != self.source_num:
        print("Error: The model has " +str(self.source_num) +" sources. Please specify adaptive_alpha for every source")
        return

    # Add additional basis for feature learning (semi-supervised NMF)
    if additional_basis>0:
      self.W_cluster=np.append(self.W_cluster, np.ones(additional_basis)*self.source_num)
      self.source_num=self.source_num+1
      self.basis_num=self.basis_num+additional_basis
      adaptive_alpha = np.append(adaptive_alpha, 1)
    
    # supervised NMF
    W, H = NMF_init(data, self.basis_num, init='random')
    W[:, 0:self.basis_num-additional_basis] = self.W

    violation=0
    Ht=H.T
    Ht = check_array(H.T, order='C')
    X = check_array(data, accept_sparse='csr')
    
    W = np.array(W)
    for run in range(iter):
      prev_W = np.array(W)
      # update H
      violation += basis_update(X.T, W=Ht, Ht=W, l1_reg=0, l2_reg=0, shuffle=False, random_state=None)
      # update W
      violation += basis_update(X, W=W, Ht=Ht, l1_reg=0, l2_reg=0, shuffle=False, random_state=None)
      
      for i in range(self.source_num):
        index = np.where(self.W_cluster == i)
        if adaptive_alpha[i] == 0:
          W[:,index] = prev_W[:,index]
        else:
          W[:,index] = prev_W[:,index]*(1-adaptive_alpha[i]) + W[:,index]*adaptive_alpha[i]

    self.W=W
    self.H=Ht.T
    self.nmf_output(input_data, self.time_vec, baseline)
    #self.time_vec=self.time_vec[:,0]

  def save_model(self, filename='NMF_model.mat', folder_id=[]):
    #import save_parameters
    nmf_model=save_parameters()
    nmf_model.supervised_nmf(self.f, self.W, self.W_cluster, self.source_num, self.feature_length, self.basis_num, self.H)
    savemat(filename, {'save_nmf':nmf_model})
    print('Successifully save to '+filename)
    
    # save the result in Gdrive as a mat file
    if folder_id:
      Gdrive=gdrive_handle(folder_id)
      Gdrive.upload(filename)
      
  def model_check(self, model):
    print('Model parameters check')
    intf=model['save_nmf']['f'].item()[0][1]-model['save_nmf']['f'].item()[0][0]
    print('Minima and maxima frequancy bin:', min(model['save_nmf']['f'].item()[0]), 'Hz and', max(model['save_nmf']['f'].item()[0]), 'Hz')
    print('Frequancy resolution:' ,intf, 'Hz')
    print('Feature length:' ,self.feature_length)
    print('Number of basis:' ,self.basis_num)
    print('Number of source:' ,self.source_num)
    if np.any(np.array(model['save_nmf'][0].dtype.names)=='sparseness'):
      print('Sparseness:', self.sparseness)
  
  def load_model(self, filename):
    model = loadmat(filename)
    self.W=model['save_nmf']['W'].item()
    self.W_cluster=model['save_nmf']['W_cluster'].item()[0]
    self.source_num=model['save_nmf']['k'].item()[0][0]
    self.feature_length=model['save_nmf']['time_frame'].item()[0][0]
    self.basis_num=model['save_nmf']['basis_num'].item()[0][0]
    self.H=model['save_nmf']['H'].item()
    if np.any(np.array(model['save_nmf'][0].dtype.names)=='sparseness'):
      self.sparseness=model['save_nmf']['sparseness'].item()[0][0]
    self.model_check(model)

