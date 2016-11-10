import librosa
import numpy as np

'''
Get the name of the original song for a path.
'''
def get_label(path):
    label = path.split('/')[2]
    return label

'''
takes the audio path and sampling rate as input, and returns the audio time series along with the sampling rate.
'''
def get_raw_time_series(path, sr = 8000):
    y, sr = librosa.load(path, sr)
    return y, sr


'''
create a tensor of the audio time series by slicing the audio time series into frames.
input y is an n-dimensional vector, window-size is an integer.
Output is a matrix/tensor of dimension window_size * floor(n/window_size).
'''
def get_audio_tensor(y, window_size = 512):
    audio_tensor = librosa.util.frame(y, window_size, window_size)
    return audio_tensor


'''
takes the audio series as input and returns the mel-scaled spectrogram along with its log-amplitudes.
both outputs are of dimension n_mels * ceil(n/window_size). window_size = hop_length in librosa lingo.
'''
def get_mel_spectogram(y, sr = 8000, window_size = 512):
    S = librosa.feature.melspectrogram(y, sr= sr, n_mels=128, hop_length = window_size)
    log_S = librosa.logamplitude(S, ref_power= np.max)
    return S, log_S


'''
takes the log-amplitude of the audio as input, and returns the Mel-frequency cepstral coefficients.
output is of dimension n_mfcc * log_S.shape[1].
'''
def get_mfcc(log_S, n_mfcc = 13):
    return librosa.feature.mfcc(S=log_S, n_mfcc=13)


def main():
    # demonstrating the functionality.
    sample_path = librosa.util.example_audio_file()
    y, sr = get_raw_time_series(sample_path)
    print('y shape', y.shape)
    print('sr', sr)
    audio_tensor = get_audio_tensor(y)
    print('audio_tensor shape', audio_tensor.shape)
    mel_spectrogram, log_amps = get_mel_spectogram(y)
    print('mel spec. shape', mel_spectrogram.shape)
    print('mel spec. log-amps shape', log_amps.shape)
    mfcc = get_mfcc(log_amps)
    print('mfcc shape', mfcc.shape)

if __name__ == "__main__":
    main()











