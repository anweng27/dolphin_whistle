import plotly.graph_objects as go
import numpy as np
import pandas as pd 
import matplotlib as plt
import matplotlib.cm as cm
import audioread




def prewhiten(input_data, prewhiten_percent):
 import numpy.matlib
 ambient = np.percentile(input_data, prewhiten_percent)
 input_data = np.subtract(input_data, np.matlib.repmat(ambient, input_data.shape[0],input_data.shape[1]))
 return input_data

# Within sample normalization
def frame_normalization(input, axis=0):
    t=np.max(input,axis=axis)
    if axis==0:
      input=input/np.matlib.repmat(np.max(input, axis=axis),input.shape[0],1)
    elif axis==1:
      input=input/np.matlib.repmat(np.max(input, axis=axis).T,input.shape[1],1).T
    input[np.isnan(input)]=0
    return input

def plot_spec(input,audio):
    #import matplotlib.pyplot as plt
    #import matplotlib.cm as cm
    #fig, ax = plt.subplots(figsize=(x_size, y_size))
    #im = ax.imshow(input, origin='lower',  aspect='auto', cmap=cm.jet,
    #                  extent=[start_time, end_time, f_range[0], f_range[-1]], interpolation='none')
    #ax.set_ylabel(y_label)
    #ax.set_xlabel(x_label)
    #cbar = fig.colorbar(im, ax=ax)
    fig = go.Figure(data=go.Heatmap(
        z=input[:,1:].T,
        x=input[:,0],
        y=audio.f/1000,
        colorscale='Jet'))
    fig.show()

      
def preprocessing(audio,plot=True,x_prewhiten=10,y_prewhiten=80,sigma=2):
    from soundscape_IR.soundscape_viewer import matrix_operation
    # Prewhiten on temporal scale 
    temp=matrix_operation.prewhiten(audio.data[:,1:], prewhiten_percent=x_prewhiten, axis=0)
    # Remove the broadband clicks(prewhiten vertically)
    temp=matrix_operation.prewhiten(temp, prewhiten_percent=y_prewhiten, axis=1)
    #Smooth the spectrogram
    from scipy.ndimage import gaussian_filter
    temp=gaussian_filter(temp, sigma=sigma)
    temp[temp<0]=0
    #normalize the energy 
    temp=frame_normalization(temp, axis=1) # It may still be better to normalize the energy of each frame 
    input_data=np.hstack((audio.data[:,0:1], temp))
    # Plot the processed spectrogram
    if plot==True: 
      plot_spec(input = input_data,audio=audio)  
    print(input_data.shape)
    return input_data


class audio_processing:
  from dolphin_whistle.audio_processing import preprocessing
  import os
  def GetGdrive(self, folder_id = []):
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    from google.colab import auth
    from oauth2client.client import GoogleCredentials
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
   
    file_extension= '.wav'
    location_cmd="title contains '"+file_extension+"' and '"+folder_id+"' in parents and trashed=false"
    file_list = drive.ListFile({'q': location_cmd}).GetList()
    i = 0
    for file in file_list:
      print('file_no = ' + str(i),'%s' % (file['title']))
      i = i + 1
  
    self.filelist = file_list
    self.folderid = folder_id
    self.Drive = drive
    
  def local_max_detector(self,audio,tonal_threshold=0.5, temporal_prewhiten=25, spectral_prewhiten=25,threshold=1, smooth=1,plot=True):
    from soundscape_IR.soundscape_viewer import tonal_detection
    detector=tonal_detection(tonal_threshold=tonal_threshold, temporal_prewhiten=temporal_prewhiten, spectral_prewhiten=spectral_prewhiten)
    self.output, self.detection=detector.local_max(audio.data, audio.f, threshold=threshold, smooth=smooth)
    # Use plotly to produce an interactive plot
    if plot==True:
      fig = go.Figure(data=go.Heatmap(
        z=output[:,1:].T,
        x=output[:,0],
        y=audio.f/1000,
        colorscale='Jet'))
      fig.show()
      fig.update_layout(
          autosize=False,
          width=500,
          height=500,
          )
    return(self.output)


  def audio_concatenation(self, file_no):
    title = self.filelist[file_no]['title']
    temp= self.filelist[file_no]
    temp.GetContentFile(temp['title'])
    location_cmd="title contains '"+title[:-4]+"' and '"+self.folderid+"' in parents and trashed=false"
    file_list = self.Drive.ListFile({'q': location_cmd}).GetList()
    i = 0 
    for file1 in filelist:
      txt = file1['title'].find('txt')
      if(txt != -1):
        temp= file_list[i]
        temp.GetContentFile(temp['title'])
      i = i + 1
    audio = AudioSegment.from_wav(title)
    merge= AudioSegment.empty()  
    df = pd.read_table(temp['title'],index_col=0)   
    for i in range(len(df)) : 
      a = df.iloc[i,2]
      b = df.iloc[i,3]
      annotation = audio[a*1000:b*1000]
      merge += annotation
    os.remove(temp['title'])
    os.remove(title)
    newtitle = title[:-4]+' concatenated.wav'
    merge.export(out_f= newtitle, format="wav")
    self.title = newtitle
    return newtitle

###Prepare the spectrogram for NMF to learn features: 
#If only want to use one audio file in the folder, input a specific 'file_no'
#If want to process all files in folder, input (file_no=None)
#type of method for preprocessing data: 1-normal preprocessing 2- local max detector
#if data is annotated and needs to be concatenated: (annotated=True)
  def prepare_spectrogram(self, preprocess_type=2, file_no = None, annotated=False,f_range=[5000,25000],plot_type='Spectrogram',time_resolution = 0.025, window_overlap=0.5,vmin=None,vmax=None, FFT_size=512,
                         tonal_threshold=0.5, temporal_prewhiten=25, spectral_prewhiten=25,threshold=1, smooth=1,plot=True,x_prewhiten=10,y_prewhiten=80,sigma=2):
    from soundscape_IR.soundscape_viewer import audio_visualization
    total_duration=0
    #if a file number is specified 
    if file_no != None: 
      if annotated:
        audio = audio_visualization(self.audio_concatenation(file_no), plot_type=plot_type,
              time_resolution=time_resolution, window_overlap=window_overlap, f_range=f_range, vmin=vmin, vmax=vmax,FFT_size=FFT_size)
      else: 
        title = self.filelist[file_no]['title']
        temp= self.filelist[file_no]
        temp.GetContentFile(temp['title'])
        audio = audio_visualization(title, plot_type=plot_type,time_resolution=time_resolution, window_overlap=window_overlap, f_range=f_range, vmin=vmin, vmax=vmax,FFT_size=FFT_size)
      self.duration = audio.data[-1,0]-audio.data[0,0]
      print(audio.data.shape)
      if preprocess_type==1:
        processed_spec=preprocessing(audio, plot=plot,x_prewhiten=x_prewhiten,y_prewhiten=y_prewhiten,sigma=sigma)
      if preprocess_type==2:
        processed_spec=local_max_detector(audio,tonal_threshold=tonal_threshold, temporal_prewhiten=temporal_prewhiten, spectral_prewhiten=spectral_prewhiten,threshold=threshold, smooth=smooth,plot=plot)
      self.data = processed_spec
      self.temp=audio.data

    #if all files in the entire folder shall be processed      
    else:
      for j in range(0,len(self.filelist)):
        if annotated: 
          audio = audio_visualization(self.audio_concatenation(j), plot_type=plot_type,
              time_resolution=time_resolution, window_overlap=window_overlap, f_range=f_range, vmin=vmin, vmax=vmax,FFT_size=FFT_size)
        else: 
          title=self.filelist[j]['title']
          temp= self.filelist[j]
          temp.GetContentFile(title)
          audio = audio_visualization(title, plot_type=plot_type,
              time_resolution=time_resolution, window_overlap=window_overlap, f_range=f_range, vmin=vmin, vmax=vmax,FFT_size=FFT_size)
        if preprocess_type==1:
          processed_spec = preprocessing(audio, plot=plot,x_prewhiten=x_prewhiten,y_prewhiten=y_prewhiten,sigma=sigma)
        if preprocess_type==2:
          processed_spec=local_max_detector(audio,tonal_threshold=tonal_threshold, temporal_prewhiten=temporal_prewhiten, spectral_prewhiten=spectral_prewhiten,threshold=threshold, smooth=smooth,plot=plot)
        total_duration += audio.data[-1,0]-audio.data[0,0]
        if j==0:
          combined = processed_spec
        else:
          combined = np.vstack((combined, processed_spec))
      #plot spectrogram of all audios in the folder 
      plot_spec(input=combined,audio=audio)
      self.data = np.array(combined) 
      self.duration = totalduration
    self.f = audio.f
    self.sf = audio.sf  
    print('Sampling rate: ',self.sf, ', Duration: ', self.duration)
    #os.remove(self.title)
      
    
###Prepare input audio files into fragments of specific lengths (for testing) and filter out empty ones 
  def prepare_testing(self,folder_id,dictionary_name,species_list=['Gg','Gm','Lh','Pc','Sa','Sl','Tt'], detection_row=200, save_H=False, vmin=None,vmax=None,feature_length=40, basis_num=36, save_id=[], length = 5, fragment_overlap=2, create_table=True, preprocess_type=2,f_range=[4000,25000],plot_type='Spectrogram',time_resolution = 0.025, window_overlap=0.5,FFT_size=256,
                      tonal_threshold=0.5, temporal_prewhiten=25, spectral_prewhiten=25,threshold=1, smooth=1,plot=True,x_prewhiten=10,y_prewhiten=80,sigma=2):
    from soundscape_IR.soundscape_viewer import audio_visualization
    #from dolphin_whistle.temp import supervised_nmf
    from soundscape_IR.soundscape_viewer import lts_maker
    df = pd.DataFrame()
    dic=np.load(dictionary_name)
    feature_num=dic.shape[1]
    audio_get=lts_maker()
    model=supervised_nmf(feature_length=feature_length, basis_num=feature_num)
    model.source_num=1
    model.W_cluster=np.ones(feature_num)*0
    model.W=dic 
    k=-1
    for species in species_list:
      audio_get.collect_Gdrive(folder_id=folder_id, file_extension=species, subfolder=True)
      audio_get.Gdrive.list_display()
      for file in audio_get.Gdrive.file_list:
        file.GetContentFile(file['title'])
        if file['title'].endswith(".wav") or file['title'].endswith(".WAV"):
          #if first time running separation and save model 
          if save_H:
            with audioread.audio_open(file['title']) as temp:
              self.duration=temp.duration
              duration = int(temp.duration)
            audio = audio_visualization(file['title'], plot_type=plot_type,time_resolution=time_resolution, window_overlap=window_overlap, f_range=f_range, vmin=vmin, vmax=vmax,FFT_size=FFT_size)
            if preprocess_type==1:
              processed_spec = preprocessing(audio, plot=plot,x_prewhiten=x_prewhiten,y_prewhiten=y_prewhiten,sigma=sigma)
            if preprocess_type==2:
              processed_spec=self.local_max_detector(audio,tonal_threshold, temporal_prewhiten, spectral_prewhiten,threshold,smooth,plot)
            model.supervised_separation(processed_spec, audio.f, iter=50)
            model.save_model(self.output, self.detection.to_numpy(), filename=file['title']+'.mat', folder_id=save_id)
            self.data = processed_spec
            self.f = audio.f
            print(model.H.shape)
          else: 
            #if there already exists a mat file for separation data
            model.load_model(file['title']+'.mat') 
          if model.H.shape[1]> length*model.feature_length: 
            for i in range(0, model.H.shape[1], (length-fragment_overlap)*model.feature_length):
              if i+length*model.feature_length < model.H.shape[1]:
                detection=model.detection[(model.detection[:,0] >= (i/model.feature_length)) & (model.detection[:,0]<=(i/model.feature_length+length))]
                print(len(detection[:,0]))
                if(len(detection[:,0])>detection_row):
                  H=model.H[:,i:i+length*model.feature_length]
                  if create_table==True:
                    k=k+1
                    sum =  np.sum(H,axis=1)
                    #sum = np.sum(H>0.01,axis=1)
                    arr = np.array([k,species,file['title'],i/model.feature_length,len(detection[:,0])])
                    stacked = np.hstack((arr.reshape(1,-1),sum.reshape(1,-1)))
                    print(k,file['title'],i/model.feature_length,len(detection[:,0]))
                    df2 = pd.DataFrame(stacked)
                    df= df.append(df2)
                else: print('not detected',file['title'],i/model.feature_length)
              else: print('last fragment too short')
    df=df.rename(columns={0:'Num',1: 'Species', 2: 'File',3:'Time',4:'detection rows'})
    self.df=df 
    self.H=model.H   
    #plot_spec(input=combined,audio=audio) 
  
