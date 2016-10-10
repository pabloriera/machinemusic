import numpy as np
from scipy import linalg
from sklearn.utils import as_float_array
from sklearn.base import TransformerMixin, BaseEstimator


def normalize(_x, axis=0, epsilon = 0.01):
    
    x = _x.copy()
    
    if axis==0:
        
        x-=x.mean(axis)
        x/=x.std(axis)
    
    elif axis==1:
    
        x = ( x.T - x.mean(axis) ).T
        x = ( x.T / x.std(axis) ).T
        
    return ZCA(regularization=epsilon).fit(x).transform(x)


class Descriptor():
    
    function = None
    data = None
    params = None
    data = None
    
    def __init__(self,data):
        self.name = data['name']
        self.params = data['params']
        self.function = data['function']
        
    def set_func(self,func):
        self.function=func
        
    def perform(self,input_data):
        self.data = self.function(self,input_data)
#         return self.descriptor

def audiofigure(Y, audio_path, sr = 44100, dpi=60, fps=4, figsize = (12,6), ylim=(-1, 1)):

    from matplotlib import animation, rc
    import matplotlib.pyplot as plt
    from IPython.display import HTML
    from base64 import encodebytes
    import subprocess
    import os
    
    fig, ax = plt.subplots(figsize=figsize)
    video_size = np.array(fig.get_size_inches())*dpi

    duration = Y.shape[0]/float(sr)
    t = np.arange(Y.shape[0])/float(sr)
    interval = 1000.0/fps
    frames = int(duration*fps)

    ax.set_xlim(( 0, duration))
    ax.set_ylim(ylim)

    line, = ax.plot([], [], lw=2)    
    ax.plot(t,Y, lw=2)

    def init():
        line.set_data([], [])
        return (line,)

 
    def animate(i):
        x = np.array([i, i])/float(fps)
        y = ylim
        
        line.set_data(x, y)
        
        return (line,)

    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=frames, interval=interval, blit=True);


    anim.save("in.mp4",codec='h264')
    plt.close()


    command = ['ffmpeg', '-i', 'in.mp4','-y', '-i', audio_path ,'-c:v', 'libx264', '-c:a', 'libvorbis', 
               '-shortest', 'out.mp4']

    proc = subprocess.Popen(command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          stdin=subprocess.PIPE)
    proc.communicate()
    
    VIDEO_TAG = r'''<video {size} {options}>
      <source type="video/mp4" src="data:video/mp4;base64,{video}">
      Your browser does not support the video tag.
    </video>'''


    with open('out.mp4', 'rb') as video:
        vid64 = encodebytes(video.read())
        _base64_video = vid64.decode('ascii')
        _video_size = 'width="{0}" height="{1}"'.format(*video_size)

    options = ['controls', 'autoplay']
    
    #os.remove("in.mp4")
    #os.remove("out.mp4")

    html = VIDEO_TAG.format(video=_base64_video, size=_video_size, options=' '.join(options))
    return HTML(html)

class ZCA(BaseEstimator, TransformerMixin):

    def __init__(self, regularization=10**-5, copy=False):
        self.regularization = regularization
        self.copy = copy

    def fit(self, X, y=None):
        X = as_float_array(X, copy = self.copy)
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        sigma = np.dot(X.T,X) / X.shape[1]
        U, S, V = linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1/np.sqrt(S+self.regularization)))
        self.components_ = np.dot(tmp, U.T)
        return self

    def transform(self, X):
        X_transformed = X - self.mean_
        X_transformed = np.dot(X_transformed, self.components_.T)
        return X_transformed
    
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