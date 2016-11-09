import numpy as np

def lrelu(x, leak=0.2, name="lrelu"):
    import tensorflow as tf
    """Leaky rectifier.
    Parameters
    ----------
    x : Tensor
        The tensor to apply the nonlinearity to.
    leak : float, optional
        Leakage parameter.
    name : str, optional
        Variable scope to use.
    Returns
    -------
    x : Tensor
        Output of the nonlinearity.
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
    
class Data():
    
    def __init__(self,data):
        self.data = data
        self.batch_ix = 0
        self.length = self.data.shape[0]
        self.ixs = np.arange(self.length)
        
    def next_batch(self,batch_size):
        np.random.shuffle(self.ixs)
        output = self.data[self.ixs[np.arange(self.batch_ix,self.batch_ix+batch_size) % self.length]]
        self.batch_ix+=batch_size
        return output

def wav2audio_segment(filename,t1=0,t2=None):
    from scipy.io import wavfile
    import numpy as np
    fs,x = wavfile.read(filename)
    
    t1 = t1*fs
    if t2 is None:
        t2 = x.size
    else:
        t2 = t2*fs
        
    x = x[t1:t2]
    return fs,np.float64(x)/2**15
    
def audio2spectral(x,orig_fs=44100,resample_fs=22050,representation='STFT',normalize=True,magnitude=True,units='lineal',nfft_size= 2**10,nfft_hop=None,
             frame_size=64, step_size=None, n_bins = 84,normalization_axis=None):
    
    from scipy.signal import resample
    import librosa
    import numpy as np

    # Read wav file to floating values
   
    fs = resample_fs
    x = resample(x, int(x.size*fs/orig_fs))
       
    # Peak Normalization
    x/=abs(x).max()

    if step_size is None:
        step_size = int(frame_size/2)
        if step_size==0:
            step_size=1
            
    if nfft_hop is None:
        nfft_hop = int(nfft_size/2)
        
    
    if representation=='STFT':
        # STFT
        S = librosa.stft(x,n_fft=nfft_size,hop_length=nfft_hop,win_length=nfft_size )/2/nfft_size

    elif representation=='CQT':
        # CQT
        S = librosa.cqt(x,sr=fs,hop_length=nfft_hop,fmin=40.0,n_bins=n_bins,real=False)

    S = S[::-1,:]
    
    if magnitude:
        S = abs(S)
        
    if units=='db':
        S = abs(S)        
        S = 20*np.log10(S/S.max()).clip(-60,0)

    if normalize:
        S = (S - S.min(normalization_axis)) /(S.max(normalization_axis) - S.min(normalization_axis))
    
    n_frames = int( (S.shape[1]-frame_size)/step_size+1 )
       
    return np.array( [S[:,i*step_size:i*step_size+frame_size] for i in range(n_frames)] )

def montage(images):
    """Draw all images as a montage separated by 1 pixel borders.

    Parameters
    ----------
    images : numpy.ndarray
        Input array to create montage of.  Array should be:
        batch x height x width x channels.
    
    Returns
    -------
    m : numpy.ndarray
        Montage image.
    """
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))
    if len(images.shape) == 4 and images.shape[3] == 3:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1, 3)) * 0.5
    else:
        m = np.ones(
            (images.shape[1] * n_plots + n_plots + 1,
             images.shape[2] * n_plots + n_plots + 1)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                m[1 + i + i * img_h:1 + i + (i + 1) * img_h,
                  1 + j + j * img_w:1 + j + (j + 1) * img_w] = this_img
    return m