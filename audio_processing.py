import numpy as np
def prewhiten(input_data, prewhiten_percent):
 import numpy.matlib
 ambient = np.percentile(input_data, prewhiten_percent)
 input_data = np.subtract(input_data, np.matlib.repmat(ambient, input_data.shape[0],input_data.shape[1] ))
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

def plot_spec(input,f_range, x_size=20, y_size=6, start_time=0, end_time=60,x_label='Frequency',y_label='Time'):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    fig, ax = plt.subplots(figsize=(x_size, y_size))
    im = ax.imshow(input, origin='lower',  aspect='auto', cmap=cm.jet,
                      extent=[start_time, end_time, f_range[0], f_range[-1]], interpolation='none')
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    cbar = fig.colorbar(im, ax=ax)
      
def preprocessing(audio,plot=True):
    from soundscape_IR.soundscape_viewer import matrix_operation
    # Prewhiten on temporal scale 
    temp=matrix_operation.prewhiten(audio.data[:,1:], prewhiten_percent=10, axis=0)
    # Remove the broadband clicks(prewhiten vertically)
    temp=matrix_operation.prewhiten(temp, prewhiten_percent=80, axis=1)
    #Smooth the spectrogram
    from scipy.ndimage import gaussian_filter
    temp=gaussian_filter(temp, sigma=2)
    temp[temp<0]=0
    #normalize the energy 
    temp=frame_normalization(temp, axis=1) # It may still be better to normalize the energy of each frame 
    input_data=np.hstack((audio.data[:,0:1], temp))
    # Plot the processed spectrogram
    if plot==True: 
      plot_spec(input = input_data[:,1:].T,x_size=20,y_size=6,start_time=0,end_time=audio.data[-1,0]-audio.data[0,0],f_range=audio.f)  
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


  def audio_concatenation(self, file_no):
    import os
    import numpy as np
    from pydub import AudioSegment
    import pandas as pd
    title = self.filelist[file_no]['title']
    temp= self.filelist[file_no]
    temp.GetContentFile(temp['title'])
    location_cmd="title contains '"+title[:-4]+"' and '"+self.folderid+"' in parents and trashed=false"
    file_list = self.Drive.ListFile({'q': location_cmd}).GetList()
    i = 0
    for file1 in file_list:
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


  
  def prepare_spectrogram(self, file_no = None, f_range=[5000,25000],plot_type='Spectrogram',time_resolution = 0.025, window_overlap=0.5,vmin=None,vmax=None, FFT_size=512):
    from soundscape_IR.soundscape_viewer import audio_visualization
    import matplotlib as plt
    import matplotlib.cm as cm
    import os
    totalduration = 0 

    if file_no != None: 
      audio = audio_visualization(self.audio_concatenation(file_no), plot_type=plot_type,
              time_resolution=time_resolution, window_overlap=window_overlap, f_range=f_range, vmin=vmin, vmax=vmax,FFT_size=FFT_size)
      self.duration = audio.data[-1,0]-audio.data[0,0]
      print(audio.data.shape)
      processed_spec=preprocessing(audio)
      self.data = processed_spec
      self.f = audio.f
      self.sf = audio.sf
      os.remove(self.title)
      
      

    else:
      for j in range(0,len(self.filelist)):
        audio = audio_visualization(self.audio_concatenation(j), plot_type=plot_type,
              time_resolution=time_resolution, window_overlap=window_overlap, f_range=f_range, vmin=vmin, vmax=vmax,FFT_size=FFT_size)
        processed_spec = preprocessing(audio)
        print(audio.sf)
        totalduration += audio.data[-1,0]-audio.data[0,0]
        if j==0:
          combined = processed_spec
        else:
          combined = np.vstack((combined, processed_spec))
        #print(combined.shape)
        os.remove(self.title) 
      self.data = np.array(combined) 
      self.f = audio.f
      self.sf = audio.sf  
      self.duration = totalduration
      plot_spec(input=combined[:,1:].T,x_size=30,y_size=8,start_time=0,end_time=self.duration, f_range=audio.f)
      
