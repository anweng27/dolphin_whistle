import numpy as np

def cluster(data,explained_var):
   from soundscape_IR.soundscape_viewer import clustering
   cluster_result=clustering(k=explained_var, pca_percent=0.9, method='kmeans')
   cluster_result.run(input_data=data,f=np.arange(1,data.shape[1]))
   print(cluster_result.cluster)
   
   return(cluster_result.cluster)
# Within sample normalization

def frame_normalization(input, axis=0):
   t=np.max(input,axis=axis)
   if axis==0:
      input=input/np.matlib.repmat(np.max(input, axis=axis),input.shape[0],1)
   elif axis==1:
      input=input/np.matlib.repmat(np.max(input, axis=axis).T,input.shape[1],1).T
      input[np.isnan(input)]=0
   return input



class feature_reduction():
   def __init__(self, input, method):
     W = frame_normalization(input.W, axis=0)
     self.W=W
     #H = input.H
     f = input.f
     self.feature_length = input.feature_length
     self.basis_num = input.basis_num
     if method == 'freq':
       self.freq_cluster(W,f)

     elif method == 'fft':
       self.fft_cluster(W)

     elif method == 'h_seq':
       self.reshaped=self.matrix_reshape(W,f)

   def nmf_reduction(self, f, basis_num2=10, max_iter=200, beta=2, sW=None, sH=None, kernel='gpu'):
     model2=nmf_gpu(1,basis_num2)
     model2.feature_learning(self.reshaped, f, max_iter, beta, sW, sH, kernel)
     model2.plot_nmf(plot_type='Both')
     self.h2=model2.H
     self.basis_num2 = basis_num2
     self.h_seq_cluster(model2.H,1)


   #reshape matrix to run NMF
   def matrix_reshape(self,W,f):
     reshaped_w = W.T.reshape((-1,len(f)))
     basis_vector = np.arange(1, self.basis_num+1).repeat(self.feature_length)
     print(basis_vector)
     reshaped = np.hstack((basis_vector[:,None],reshaped_w))
     print(reshaped.shape)
     self.reshaped=reshaped
     return reshaped

   def freq_cluster(self,W,f):
     reshaped=self.matrix_reshape(W,f)
     for i in range(1,self.basis_num+1):
        if i==1:
           freq_distribution = np.mean(reshaped[reshaped[:,0]==i,1:],axis=0)
        else:
           freq_distribution = np.vstack((freq_distribution,np.mean(reshaped[reshaped[:,0]==i,1:],axis=0)))
     print(freq_distribution.shape)
     self.result = freq_distribution

   def fft_cluster(self,W):
      import scipy
      from scipy.fft import fft
      import matplotlib.pyplot as plt
      (start,end)=(int(self.feature_length-(self.feature_length/4)),int(self.feature_length+(self.feature_length/4)))
      for i in range(self.basis_num):
        if i==0:
           matrix = np.log(np.abs(fft(W[:,i])[start:end]))
        else:
           matrix = np.vstack((matrix,np.log(np.abs(fft(W[:,i])[start:end]))))
        plt.plot(np.log(np.abs(fft(W[:,i])[start:end])))
      print(matrix.shape)
      self.result = matrix

   def h_seq_cluster(self,H,feature_length):
     h=H[:,int(np.floor(feature_length/2)):self.feature_length*self.basis_num+int(np.floor(feature_length/2))]
     final = np.zeros(h.shape[0])
     for i in range(1,self.basis_num+1):
       max_loc= np.argmax(h[:,self.reshaped[:,0]==i],axis=1)
       a = np.sum(h[:,self.reshaped[:,0]==i], axis=1)
       index = a.argsort()[::-1]
       b = np.sum(a)
       answer = np.where(np.cumsum(np.sort(a)[::-1])>b*0.98)[0][0]
       max_loc[index[answer+1:]]=0
       t = abs(np.subtract(max_loc,50))
       t[np.where(max_loc==0)]=0
       t_index = np.argsort(t)[::-1]
       count = np.arange(self.basis_num2)[::-1]
       for j in range(0,answer+1):
         t[t_index[j]] = count[j]
       final = np.vstack((final,t))
     print(final[1:,])
     self.result = final[1:,]

   def cluster(self, explained_var, pca_percent=0.9, method='kmeans'):
     from soundscape_IR.soundscape_viewer import clustering
     cluster_result=clustering(k=explained_var, pca_percent=pca_percent, method=method)
     cluster_result.run(input_data=self.result, f=np.arange(1,self.result.shape[1])) #f=np.arange(1,self.result.shape[1]))f=np.linspace(4000, 25000, num=111)
     print(cluster_result.cluster)
     self.cluster_result=cluster_result.cluster
     self.cluster_object=cluster_result


   def view_clusters(self,cluster_num):
     self.cluster_object.plot_cluster_feature(cluster_no=cluster_num, freq_scale='linear', f_range=[], fig_width=12, fig_height=6)


   def examine_result(self):
     matrix=np.zeros((np.max(self.cluster_result),7))
     for i in range(1,np.max(self.cluster_result)+1):
       species=np.floor(np.where(self.cluster_result==i)[0]/100).astype(int)
       for j in species:
         percentage = 100/len(species)
         matrix[i-1][j] += percentage
     plot_spec(matrix,[0,np.max(self.cluster_result)],10,10,0.5,7.5,x_label='Species',y_label='Cluster')
