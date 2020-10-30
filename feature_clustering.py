class clustering:
  """
  Clustering of audio events
  1. Initiate clustering model
  2. Run clustering
  3. Save the features of each cluster as csv
  4. Save the clustering labels as csv
  
  Examples
  --------
  >>> cluster_result=clustering(k=0.8,method='kmeans')
  >>> input_data, f=LTS.input_selection('median')
  >>> cluster_result.run(input_data, f)
  >>> cluster_result.save_cluster_feature(filename='Cluster_feature.csv')
  >>> 
  >>> analysis_result=data_organize()
  >>> analysis_result.time_fill(time_vec=cluster_result.time_vec, data=cluster_result.cluster, header='LTS_Median_Cluster')
  >>> analysis_result.save_csv(filename='Cluster_result.csv')
  >>>
  >>> analysis_result.plot_diurnal(1)
  
  Parameters
  ----------
  k : float (>0)
      Number of clusters (when k>1). When k <1, algorithm will search an optimal 
      cluster number so that the clustering result can explan k of data dispersion
      Default: 2
      
  method : 'kmeans'
      Clustering method.
      Default: 'kmeans'
  
  """
  def __init__(self, k=2, method='kmeans'):
    self.k=k
    self.method=method

  def run(self, input_data, f, standardization=[]):
    self.time_vec=input_data[:,0]
    self.f=f
    input_data=input_data[:,1:]

    # standardization
    if standardization=='max-min':
      input_data=input_data-np.matlib.repmat(input_data.min(axis=1), input_data.shape[1],1).T
      input_data=np.divide(input_data, np.matlib.repmat(input_data.max(axis=1), input_data.shape[1], 1).T)
    

    input_data[np.isnan(input_data)]=0
    if self.method=='kmeans':
      cluster=self.run_kmeans(input_data)

    # extract scene features by percentiles
    soundscape_scene = np.zeros((np.max(cluster)+1,), dtype=np.object)
    scene = np.zeros((len(f),np.max(cluster)+2), dtype=np.object)
    scene[:,0]=f
    scene_label = np.zeros((np.max(cluster)+2,), dtype=np.object)
    scene_label[0] = 'Frequency (Hz)'
    for c in np.arange(np.max(cluster)+1):
      soundscape_scene[c] = np.percentile(input_data[cluster==c,:], [5, 25, 50, 75, 95], axis=0).T
      scene[0:len(f),c+1]=soundscape_scene[c][:,2]
      scene_label[c+1] = 'Scene'+ str(c+1)
 
    self.cluster=cluster+1
    self.soundscape_scene=soundscape_scene
    self.scene_feature = pd.DataFrame(scene, columns = scene_label)
    
  def run_kmeans(self, data):
    if self.k<1:
      interia=[]
      k_try=1
      while k_try:
        # Create a kmeans model on our data, using k clusters.
        kmeans_model = KMeans(n_clusters=k_try, init='k-means++').fit(data)

        # Measuring the explained variation 
        interia.append(kmeans_model.inertia_)
        print("k:",k_try, ", explained variation:", 1-interia[k_try-1]/interia[0])
        if 1-interia[k_try-1]/interia[0] >= self.k:
          k_final=k_try
          break
        else:
          k_try=k_try+1
    else:
      k_final=round(self.k
    
                    


def cluster(data,explained_var):
   import numpy as np
   from dolphin_whistle.feature_clustering import clustering
   cluster_result=clustering(k=explained_var,method='kmeans')
   cluster_result.run(input_data=data,f=np.arange(1,data.shape[1]))
   print(cluster_result.cluster)
   
   return(cluster_result.cluster)
# Within sample normalization

def frame_normalization(input, axis=0):
   import numpy as np
   t=np.max(input,axis=axis)
   if axis==0:
      input=input-np.matlib.repmat(np.min(input, axis=axis),input.shape[0],1)
      input=input/np.matlib.repmat(np.max(input, axis=axis),input.shape[0],1) 
   elif axis==1:
      input=input-np.matlib.repmat(np.min(input, axis=axis).T,input.shape[1],1).T
      input=input/np.matlib.repmat(np.max(input, axis=axis).T,input.shape[1],1).T
      input[np.isnan(input)]=0
   return input


class feature_reduction():
   import numpy as np
   def __init__(self, input, method, umap = False,neighbors=4):
     W = frame_normalization(input.W, axis=0)
     self.W=W
     #H = input.H
     f = input.f
     self.feature_length = input.feature_length
     self.basis_num = input.basis_num
     self.umap = umap
     if method == 'freq':
       self.freq_cluster(W,f)
       if umap == True:
         self.map_func(num_neighbors=neighbors)

     elif method == 'fft':
       self.fft_cluster(W)
       if umap == True:
         self.umap_func(num_neighbors=neighbors)

     elif method == 'h_seq':
       self.reshaped=self.matrix_reshape(W,f)
       if umap == True:
         self.umap_func(neighbors)
         
     elif method == 'autocorrelation':
       self.autocorrelation_cluster(W)
       if umap == True:
         self.umap_func(neighbors)

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
      self.result=matrix
      #self.result = frame_normalization(matrix,axis=1)
     
   
   def autocorrelation_cluster(self,W):
      import matplotlib.pyplot as plt
 
      for i in range(self.basis_num):
        if i==0:
           matrix = np.correlate(W[:,i],W[:,i], mode='full')
        else:
           matrix = np.vstack((matrix,np.correlate(W[:,i],W[:,i], mode='full')))
      print(matrix.shape)
      self.result=matrix


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


   def umap_func(self,num_neighbors):
     import umap
     Reduced = umap.UMAP(n_neighbors=num_neighbors).fit_transform(self.result)
     self.result = Reduced

   def cluster(self, explained_var, method='kmeans',plot=False):
     from dolphin_whistle.feature_clustering import clustering
     cluster1=clustering(k=explained_var,method=method)
     cluster1.run(input_data=np.hstack((np.arange(self.result.shape[0])[None].T, self.result)), f=np.arange(2)) #f=np.arange(1,self.result.shape[1]))f=np.linspace(4000, 25000, num=111)
     print(cluster1.cluster)
     if plot==True:
       for n in range(1, np.max(self.cluster_result)+1):
         model.plot_nmf(plot_type='W',W_list=np.where(self.cluster_result==n)[0])
     self.cluster_result=cluster1.cluster
     self.cluster_object=cluster1
     
   
   def cluster2(self,explained_var,umap = True,num_neighbors=4,plot=False, method='kmeans'):
     import umap
     #second layer cluster
     from dolphin_whistle.feature_clustering import clustering
     cluster2_num=np.zeros(np.max(autocorrelation.cluster_result))
     temp = np.zeros(self.cluster_result.shape)
     for n in range(1,np.max(self.cluster_result)+1): #looping through first layer cluster
       if self.umap == True:
         Reduced2 = umap.UMAP(n_neighbors=num_neighbors).fit_transform(model.W[:,np.where(self.cluster_result==n)[0]].T)
       else: 
         Reduced2 = Reduced
       cluster2=clustering(k=explained_vard)
       cluster2.run(np.hstack((np.arange(Reduced2.shape[0])[None].T, Reduced2)), f=np.arange(2))
       cluster2_num[n-1]=np.max(cluster2.cluster)
       for j in range(1, np.max(cluster2.cluster)+1):
         temp[np.where(self.cluster_result==n)[0][np.where(cluster2.cluster==j)[0]]] = n*100+j
       import matplotlib.pyplot as plt
       plt.scatter(Reduced2[:,0], Reduced2[:,1], c=cluster2.cluster)
       plt.show()
       if plot==True:
         for j in range(1, np.max(cluster2.cluster)+1):
           model.plot_nmf(plot_type='W',W_list=np.where(self.cluster_result==n)[0][np.where(cluster2.cluster==j)[0]])
       print(cluster2_num)
       self.temp=temp
       self.cluster2_result=cluster2.cluster
       self.cluster2_object=cluster2
       self.cluster2_num=cluster2_num

   def view_clusters(self,cluster_num,cluster_layer=2):
     if cluster_layer==1:
       self.cluster_object.plot_cluster_feature(cluster_no=cluster_num, freq_scale='linear', f_range=[], fig_width=12, fig_height=6)
     if cluster_layer==2:
       self.cluster2_object.plot_cluster_feature(cluster_no=cluster_num, freq_scale='linear', f_range=[], fig_width=12, fig_height=6)


   def generate_heatmap(self,cluster_layer=2):
     if cluster_layer==1:
       matrix=np.zeros((np.max(self.cluster_result),7))
       for i in range(1,np.max(self.cluster_result)+1):
         species=np.floor(np.where(self.cluster_result==i)[0]/100).astype(int)
         for j in species:
          percentage = 100/len(species)
          matrix[i-1][j] += percentage
       plot_spec(matrix,[0,np.max(self.cluster_result)],10,10,0.5,7.5,x_label='Species',y_label='Cluster')
  
     if cluster_layer==2:
       matrix=np.zeros((np.sum(self.cluster2_num).astype(int)+1,7+2))
       cumsum=np.cumsum(self.cluster2_num).astype(int)
       cumsum = np.insert(cumsum, 0, 0, axis=0)
       print(cumsum)
       for k in range(1,np.max(self.cluster_result)+1):
         for j in range(1,self.cluster2_num[k-1].astype(int)+1):
           species=np.floor(np.where(self.temp==k*100+j)[0]/100).astype(int)
           #print(species)
           matrix[cumsum[k-1].astype(int)+(j)][-2]=k
           matrix[cumsum[k-1].astype(int)+(j)][-1]=j
           for t in species:
            percentage = 100/len(species)
            matrix[cumsum[k-1].astype(int)+(j)][t] += percentage
       self.matrix=matrix
       #print(matrix)
       #plot_spec(matrix,[0,np.sum(cluster2_num)],10,10,0.5,7.5,x_label='Species',y_label='Cluster')
       #Assign figure size
       fig, ax = plt.subplots(figsize=(200,40))
       im = ax.imshow(matrix[1:,:-2], cmap = 'jet', interpolation = None, vmin = 0, vmax = 100)
       x_label_list= ['Gg','Gm','Lg','Pc','Sa','Sl','Tt']
       ax.set_xticks([0,1,2,3,4,5,6])
       ax.set_xticklabels(x_label_list)
       ax.set_yticks(np.arange(matrix.shape[0]))
       ax.set_yticklabels(1+np.arange(matrix.shape[0])[:-1])
       #Assign fontsize
       ax.tick_params(labelsize = 18)
       ax.set_xlabel('Species', fontsize = 20)
       ax.set_ylabel('Cluster', fontsize = 20)
       cbar = fig.colorbar(im, ax=ax)
       cbar.ax.tick_params(labelsize = 18)
       cbar.set_label('Percentage', fontsize = 2)

   def check_cluster(self,num):
     print(self.matrix[num,:-2])
     (a,b)=(self.matrix[num,-2].astype(int),self.matrix[num,-1].astype(int))
     print(a,b)
     model.plot_nmf(plot_type='W',W_list=np.where(self.temp==a*100+b)[0])
     print(np.floor(np.where(self.temp==a*100+b)[0]/100).astype(int))
