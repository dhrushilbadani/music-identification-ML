import os
from fnmatch import fnmatch
from audio_processing import get_label, get_raw_time_series, get_audio_tensor, get_mfcc, get_mel_spectogram
import pickle
import numpy as np
import sys

OUTER_DIRECTORY = 'coversongs/covers32k'

'''
Recursively iterates through OUTER_DIRECTORY and returns a list of all mp3 filepaths.
Also returns a list of all original song names.
'''
def get_data():
    try:
        allfiles = pickle.load(open('serialized/all_filenames.pkl', "rb"))
        song_name_to_id = pickle.load(open('serialized/song_name_to_id.pkl', "rb"))
        id_to_song_name = pickle.load(open('serialized/id_to_song_name.pkl', "rb"))
        original_song_names = pickle.load(open('serialized/original_song_names.pkl', "rb"))
    except:
        allfiles = []
        original_song_names = set()
        song_name_to_id = dict()
        id_to_song_name = dict()
        i = 0
        for dirpath, dirnames, filenames in os.walk(OUTER_DIRECTORY):
            for filename in [f for f in filenames if f.endswith(".mp3")]:
                full_path = os.path.join(dirpath, filename)
                allfiles.append(full_path)
                original_song_name = full_path.split('/')[2]
                original_song_names.add(original_song_name)
                if original_song_name not in song_name_to_id:
                    song_name_to_id[original_song_name] = i
                    id_to_song_name[i] = original_song_name
                    i += 1
        pickle.dump(allfiles, open('serialized/all_filenames.pkl', "wb"))
        pickle.dump(original_song_names, open('serialized/original_song_names.pkl', "wb"))
        pickle.dump(song_name_to_id, open('serialized/song_name_to_id.pkl', "wb"))
        pickle.dump(id_to_song_name, open('serialized/id_to_song_name.pkl', "wb"))
    return allfiles, original_song_names, song_name_to_id, id_to_song_name


'''
Returns all the features for a song, namely:
1. Raw time series
2. The audio tensor (check audio_processing.py for more)
3. The spectrogram and log amplitudes of the spectrogram
4. The mfcc
'''
def all_features_for_song(path, sr = 8000, window_size = 512, n_mfcc = 13):
    time_series, sr = get_raw_time_series(path, sr)
    audio_tensor = get_audio_tensor(time_series, window_size)
    S, log_S = get_mel_spectogram(time_series, sr, window_size)
    mfcc = get_mfcc(log_S, n_mfcc)
    return time_series, audio_tensor.T, S.T, log_S.T, mfcc.T


'''
Helper to compute the label vector for a feature matrix for a particular song specified by path.
'''
def compute_label_vector(feature_matrix, path):
    original_song_name = path.split('/')[2]
    original_song_id = song_name_to_id[original_song_name]
    n = feature_matrix.shape[0]
    return original_song_id * np.ones((n, 1))


'''
Computes and serializes the aforementioned features for the whole dataset.
'''
def compute_features_dataset(paths):
    try:
        time_series_matrices = pickle.load(open('features/time_series_matrix.pkl', "rb"))
        audio_tensor_matrix = np.load('features/audio_tensor_matrix.npy')
        S_matrix = np.load('features/S_matrix.npy')
        log_S_matrix = np.load('features/log_S_matrix.npy')
        mfcc_matrix = np.load('features/mfcc_matrix.npy')

        time_series_labels = pickle.load(open('features/time_series_labels.pkl', "rb"))
        audio_tensor_labels_matrix = np.load('features/audio_tensor_labels_matrix.npy')
        S_labels_matrix = np.load('features/S_labels_matrix.npy')
        log_S_labels_matrix = np.load('features/log_S_labels_matrix.npy')
        mfcc_labels_matrix = np.load('features/mfcc_labels_matrix.npy')

    except:
        if not os.path.exists('features'):
            os.makedirs('features')
        time_series_matrices, audio_tensor_matrices, S_matrices, log_S_matrices, mfcc_matrices = [], [], [], [], []
        time_series_labels, audio_tensor_labels, S_labels, log_S_labels, mfcc_labels = [], [], [], [], []
        i = 0
        for path in paths:
            i += 1
            time_series, audio_tensor, S, log_S, mfcc = all_features_for_song(path)

            audio_tensor_label_vec = compute_label_vector(audio_tensor, path)
            S_label_vec = compute_label_vector(S, path)
            log_S_label_vec = compute_label_vector(log_S, path)
            mfcc_label_vec = compute_label_vector(mfcc, path)

            time_series_matrices.append(time_series)
            S_matrices.append(S)
            log_S_matrices.append(log_S)
            mfcc_matrices.append(mfcc)
            audio_tensor_matrices.append(audio_tensor)

            original_song_name = path.split('/')[2]
            original_song_id = song_name_to_id[original_song_name]
            time_series_labels.append(original_song_id)
            S_labels.append(S_label_vec)
            log_S_labels.append(log_S_label_vec)
            mfcc_labels.append(mfcc_label_vec)
            audio_tensor_labels.append(audio_tensor_label_vec)

            if (i % 4 == 0):
                print(str(i) + ' completed.')


        audio_tensor_matrix = np.vstack(audio_tensor_matrices)
        S_matrix = np.vstack(S_matrices)
        log_S_matrix = np.vstack(log_S_matrices)
        mfcc_matrix = np.vstack(mfcc_matrices)

        pickle.dump(time_series_matrices, open('features/time_series_matrix.pkl', "wb"))
        audio_tensor_matrix.dump('features/audio_tensor_matrix.npy')
        S_matrix.dump('features/S_matrix.npy')
        log_S_matrix.dump('features/log_S_matrix.npy')
        mfcc_matrix.dump('features/mfcc_matrix.npy')
        print('Finished calculating and dumping features...')

        del time_series_matrices, audio_tensor_matrix, S_matrix, log_S_matrix, mfcc_matrix
        # freeing up memory

        audio_tensor_labels_matrix = np.vstack(audio_tensor_labels)
        S_labels_matrix = np.vstack(S_labels)
        log_S_labels_matrix = np.vstack(log_S_labels)
        mfcc_labels_matrix = np.vstack(mfcc_labels)

        pickle.dump(time_series_labels, open('features/time_series_labels.pkl', "wb"))
        audio_tensor_labels_matrix.dump('features/audio_tensor_labels_matrix.npy')
        S_labels_matrix.dump('features/S_labels_matrix.npy')
        log_S_labels_matrix.dump('features/log_S_labels_matrix.npy')
        mfcc_labels_matrix.dump('features/mfcc_labels_matrix.npy')
        print('Finished calculating and dumping labels...')
    print('Done!')


all_music_files, original_song_names, song_name_to_id, id_to_song_name = get_data()
compute_features_dataset(all_music_files)