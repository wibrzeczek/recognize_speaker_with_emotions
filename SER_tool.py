#-------------------------------------IMPORT BIBLIOTEK--------------------------------
import os
import sys
import h5py
import json
import keras
import pickle
import librosa
import sklearn
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import layers
import noisereduce as nr
from keras import callbacks 
from librosa import display 
from keras import optimizers  
import IPython.display as ipd
import matplotlib.pyplot as plt 
from json_tricks import dump, load
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM, Dense
from pydub import AudioSegment, effects
from keras.models import model_from_json
from sklearn.metrics import confusion_matrix
from noisereduce.noisereducev1 import reduce_noise  
from sklearn.model_selection import train_test_split


  

# Uruchom kod i przechwyć wszystkie ostrzeżenia
with warnings.catch_warnings():
  warnings.filterwarnings("ignore")
  
# Uruchom program, w którym będą generowane ostrzeżenia
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 


#--------------------------------WSTĘPNE PRZETWARZANIE DLA JEDNEGO PLIKU------------------------
#Zdefiniowana ścieżka dla pojedynczego pliku
os.chdir('/Users/Admin/Desktop/')
path = "../Desktop/train_gender/female_test/1002_IWL_NEU_XX.wav"


# Ładowanie pliku do Audio Segment
# Definiowanie częstotliwość próbkowania.

rawsound = AudioSegment.from_file(path)
x, sr = librosa.load(path, sr = None)

plt.figure(figsize=(15,1))
librosa.display.waveshow(x, sr)
plt.title('Próbka audio')
plt.show()


normsound = effects.normalize(rawsound, headroom = 5.0) 
normalized_x = np.array(normsound.get_array_of_samples(), dtype = 'float32')

plt.figure(figsize=(15,2))
librosa.display.waveshow(normalized_x, sr)
plt.title('Znormalizowany plik audio')
plt.show()


xtrim, index = librosa.effects.trim(normalized_x, top_db = 30)

plt.figure(figsize=(6,2))
librosa.display.waveshow(xtrim, sr)
plt.title('Próbka Trimmed')

ipd.display(ipd.Audio(data = xtrim, rate=sr))
plt.show()

#Dodanie do pliku brakujących wartości
x_pad = np.pad(xtrim, (0, 173056-len(xtrim)), 'constant')

plt.figure(figsize=(12,2))
librosa.display.waveshow(x_pad, sr)
plt.title('Uzupełniona próbka')

ipd.display(ipd.Audio(data = x_pad, rate=sr))
plt.show()

x_result = reduce_noise(audio_clip=x_pad, noise_clip=x_pad, verbose=False)

plt.figure(figsize=(12,2))
librosa.display.waveshow(x_result, sr)
plt.title('Ostateczna wersja próbki po redukcji szumu')

ipd.display(ipd.Audio(data = x_result, rate=sr))
plt.show()

#------------------------------------DEFINICJA FUNKCJI CECH DLA JEDNEGO PLIKU--------------------------------

length_of_frame = 2048
hop_length = 512

feature1 = librosa.feature.rms(y=x_result, frame_length=length_of_frame, hop_length=hop_length) # Energy - Root Mean Square (RMS)
feature2 = librosa.feature.zero_crossing_rate(y=x_result, frame_length=length_of_frame, hop_length=hop_length) # Zero Crossed Rate (ZCR)
feature3 = librosa.feature.mfcc(y=x_result, sr=sr, S=None, n_mfcc=13, hop_length = hop_length) # MFCCs

# Wyświetlenie obliczonych wartości cech
print('Enegia: ', feature1.shape)
print('ZCR: ', feature2.shape)
print('MFCCs: ', feature3.shape)

def def_emotion(name): 
        if('NEU' in name): return "01"
        elif('HAP' in name): return "02"
        elif('SAD' in name): return "03"
        elif('ANG' in name): return "04"
        else: return "-1"

def normalise_emotion(idx_emotion):
    if idx_emotion == "01":   
      return 0 # dla emocji neutral
    elif idx_emotion == "02": 
      return 1 # dla emocji happy
    elif idx_emotion == "03": 
      return 2 # dla emocji sad
    elif idx_emotion == "04" : 
      return 3 # dla emocji angry

length_of_sample = []
folder_path = '/Users/Admin/Desktop/Praca_inzynierska/test_neutral/'

for subdir, dirs, files in os.walk(folder_path):
  for file in files: 
    x, sr = librosa.load(path = os.path.join(subdir,file), sr = None)
    xtrim, index = librosa.effects.trim(x, top_db=30)
     
    length_of_sample.append(len(xtrim))

print('Maksymalna długość próbki:', np.max(length_of_sample))    

#----------------------------------WSTĘPNE PRZETWARZANIE DLA  PLIKU--------------------------------------

# Definiowanie wymaganych zmiennych
total_length = 173056 
length_of_frame = 2048
hop_length = 512

# Definiowane list
list_rms = []
list_zcr = []
list_mfcc = []
list_emotions = []

audio_test = '/Users/Admin/Desktop/Praca_inzynierska/thresholds/1112 (z JL corpus)/Angry/female1_angry_6b_1_ANG.wav'


_, sr = librosa.load(path = os.path.join(subdir,file), sr = None) 
rawsound = AudioSegment.from_file(os.path.join(subdir,file)) 
normsound = effects.normalize(rawsound, headroom = 0) 
normalized_x = np.array(normsound.get_array_of_samples(), dtype = 'float32')
xtrim, index = librosa.effects.trim(normalized_x, top_db=30)
x_pad = np.pad(xtrim, (0, total_length-len(xtrim)), 'constant')
x_result = nr.reduce_noise(x_pad, sr=sr) 
       
# Ekstrakcja cech dla każdego z plików 
feature1 = librosa.feature.rms(x_result, frame_length=length_of_frame, hop_length=hop_length)   
feature2 = librosa.feature.zero_crossing_rate(x_result , frame_length=length_of_frame, hop_length=hop_length, center=True)     
feature3 = librosa.feature.mfcc(x_result, sr=sr, n_mfcc=13, hop_length = hop_length) 
    
# Definiowanie emocji w bazie danych
if (def_emotion(file) != "-1"): 
    name = def_emotion(file)
    print(name)
else:                              
    name = file[6:8]                      
                
# Dodawanie danych do stworzonych list
list_rms.append(feature1)
list_zcr.append(feature2)
list_mfcc.append(feature3)
list_emotions.append(normalise_emotion(name)) 

feature_rms = np.asarray(list_rms).astype('float32')
feature_rms = np.swapaxes(feature_rms,1,2)
feature_zcr = np.asarray(list_zcr).astype('float32')
feature_zcr = np.swapaxes(feature_zcr,1,2)
feature_mfccs = np.asarray(list_mfcc).astype('float32')
feature_mfccs = np.swapaxes(feature_mfccs,1,2)


# Wyświetlenie obliczonych wartości cech dla wszystkich plików

print('ZCR:',feature_zcr.shape)
print('RMS:',feature_rms.shape)
print('MFCCs:',feature_mfccs.shape)

# Łączenie wszystkich funkcji w zmienną „X”.
X_t = np.concatenate((feature_zcr, feature_rms, feature_mfccs), axis=2)

Y_t = np.array(list_emotions)


# Zapisanie zdefiniowanych tablic X, Y jako listy do plików json.
os.chdir('/Users/Admin/Desktop/train/')

X_tt = X_t.tolist() 

with open("x_probowac_2.json", "w") as put_file:
    json.dump(X_tt, put_file)

Y_tt = Y_t.tolist() 

with open("y_probowac_2.json", "w") as put_file:
    json.dump(Y_tt, put_file)



#--------------------------------------WCZYTANIE ZAPISANEGO MODELU----------------------------------


# Ładowanie modelu z zapisanego pliku json

path_to_model = '/Users/Admin/Desktop/Praca_inzynierska/SER_program/Models/model_6_caly.json'
path_to_weights = '/Users/Admin/Desktop/Praca_inzynierska/SER_program/Weights/model_6_caly.h5'

with open(path_to_model , 'r') as json_file:
    modelsave_json = json_file.read()
    
# Ładowanie architektury modelu
model = tf.keras.models.model_from_json(modelsave_json)
model.load_weights(path_to_weights)

# Kompilacja modelu o podobnych parametrach do modelu oryginalnego
model.compile(loss='categorical_crossentropy', 
                optimizer='RMSProp', 
                metrics=['categorical_accuracy'])


#--------------------------------------TEST START------------------------------------------------------------

os.chdir('/Users/Admin/Desktop/train/')

X_t = load( 'x_probowac_2.json')
X_t = np.asarray(X_t).astype('float32')

Y_t = load('y_probowac_2.json')
Y_t  = np.asarray(Y_t).astype('int8')

Y_t_class = tf.keras.utils.to_categorical(Y_t-1, 4, dtype = 'int8')

loss, acc = model.evaluate(X_t, Y_t_class, verbose=2)


Y_t_class = np.argmax(Y_t_class, axis=1)
predictions_t = model.predict(X_t)

y_pr_class_t = np.argmax(predictions_t, axis=1)

print('Y_T_CLASS ' , Y_t_class )
print('PREDICTIONS ' , y_pr_class_t)

 #--------------------------------------TEST END-----------------------------------------------------------

#----------------------------------IMPLEMENTACJA BIBLIOTEK-----------------------------
import os
import wave
import time
import pickle
import shutil
import pyaudio
import librosa
import warnings
import soundfile
import numpy as np
import pandas as pd
import librosa.display
import seaborn as sns
import IPython.display as ipd 
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.playback import play
from sklearn import preprocessing
from IPython.display import Audio 
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings("ignore")


#--------------------  ------FUNKCJE EKSTRAKCJI CECH DLA PLIKÓW AUDIO------------------------------------------------
# Kalkulacja dodatkowego parametru dla MFCC
def calculate_delta(array):
   
    wiersz,kolumny = array.shape
    print(wiersz)
    print(kolumny)
    val_delta = np.zeros((wiersz,20))
    N = 2
    for i in range(wiersz):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
              pierwszy =0
            else:
              pierwszy = i-j
            if i+j > wiersz-1:
                drugi= wiersz-1
            else:
                drugi= i+j 
            index.append((drugi,pierwszy))
            j+=1
        val_delta[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
    return val_delta

# Funkcja ekstrakcji cech dla MFCC
def extract_MFCC(audio,rate):
       
    fea_mfcc = mfcc.mfcc(audio,rate, 0.025, 0.01,20,nfft = 1200, appendEnergy = True)    
    fea_mfcc = preprocessing.scale(fea_mfcc)
    print(fea_mfcc)
    delta = calculate_delta(fea_mfcc)
    mixted = np.hstack((fea_mfcc,delta)) 
    return mixted

# Funkcja ekstrakcji cech dla CHROMA
def extract_chroma(filename):

    y, sr = librosa.load(filename, duration=3, offset=0.5)
    with soundfile.SoundFile(filename) as sound_file:
        X = sound_file.read(dtype="float32")
    stft=np.abs(librosa.stft(X))
    chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T,axis=0)    
    return chroma

# Funkcja ekstrakcji cech dla Mel-Spektrogram
def extract_mel(filename):
 
    with soundfile.SoundFile(filename) as sound_file:
        X = sound_file.read(dtype="float32")
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mel=np.mean(librosa.feature.melspectrogram(X, sr=sr).T,axis=0)
    return mel

#---------------------------------------ZDEFINIOWANIE KLASYFIKATORÓW---------------------------------        

#Klasyfikator Złości

def test_model_angry():
    print('Klasyfikator Angry')
    os.chdir('/Users/Admin/Desktop/')
    source   = "../Desktop/train_speaker/testing_set_angry/"  
    modelpath = "../Desktop/train_speaker/trained_models_angry/"
    test_file = "../Desktop/train_speaker/testing_set_addition.txt"       
    file_paths = open(test_file,'r')
     
    gmm_files = [os.path.join(modelpath,fname) for fname in
                  os.listdir(modelpath) if fname.endswith('.gmm')]
     
    #Załadowanie modelów Gaussa
    models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
    speakers   = [fname.split("\\")[-1].split("*.gmm")[0] for fname 
                  in gmm_files]
     
    audio_path = '/Users/Admin/Desktop/BazaDanych-Ostatecznie/Emotional Speech Dataset (ESD) CREMA/1001/Angry/1001_DFA_ANG_XX.wav'

    sr,audio = read(audio_path)
    vector   = extract_MFCC(audio,sr)
            
    log_likelihood = np.zeros(len(models)) 
        
    for i in range(len(models)):
        gmm    = models[i]  # Sprawdzanie modeli (każdy model wejściowy z każdym modelem w bazie)
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()

    detected = np.argmax(log_likelihood)
    print("\tDetected as - ", speakers[detected])
    time.sleep(1.0)  

#---------------------------------------------------------------------------------------------        
#Klasyfikator Smutku
    
def test_model_sad():
    print('Klasyfikator Sad')
    os.chdir('/Users/Admin/Desktop/')
    source   = "../Desktop/train_speaker/testing_set_sad/"  
    modelpath = "../Desktop/train_speaker/trained_models_sad/"
    test_file = "../Desktop/train_speaker/testing_set_addition.txt"       
    file_paths = open(test_file,'r')
     
    gmm_files = [os.path.join(modelpath,fname) for fname in
                  os.listdir(modelpath) if fname.endswith('.gmm')]
     
    #Załadowanie modelów Gaussa
    models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
    speakers   = [fname.split("\\")[-1].split("*.gmm")[0] for fname 
                  in gmm_files]
     
    audio_path = '/Users/Admin/Desktop/BazaDanych-Ostatecznie/Emotional Speech Dataset (ESD) CREMA/1001/Angry/1001_DFA_ANG_XX.wav'

    sr,audio = read(audio_path)
    vector   = extract_MFCC(audio,sr)
            
    log_likelihood = np.zeros(len(models)) 
        
    for i in range(len(models)):
        gmm    = models[i]  # Sprawdzanie modeli (każdy model wejściowy z każdym modelem w bazie)
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
 
    detected = np.argmax(log_likelihood)
    print("\tDetected as - ", speakers[detected])
    time.sleep(1.0)  

#---------------------------------------------------------------------------------------------       
#Klasyfikator Radości

def test_model_happy():
    print('Klasyfikator Happy')
    os.chdir('/Users/Admin/Desktop/')
    source   = "../Desktop/train_speaker/testing_set_happy/"  
    modelpath = "../Desktop/train_speaker/trained_models_happy/"
    test_file = "../Desktop/train_speaker/testing_set_addition.txt"       
    file_paths = open(test_file,'r')
     
    gmm_files = [os.path.join(modelpath,fname) for fname in
                  os.listdir(modelpath) if fname.endswith('.gmm')]
     
    #Załadowanie modelów Gaussa
    models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
    speakers   = [fname.split("\\")[-1].split("*.gmm")[0] for fname 
                  in gmm_files]
     
    audio_path = '/Users/Admin/Desktop/BazaDanych-Ostatecznie/Emotional Speech Dataset (ESD) CREMA/1001/Angry/1001_DFA_ANG_XX.wav'

    sr,audio = read(audio_path)
    vector   = extract_MFCC(audio,sr)
            
    log_likelihood = np.zeros(len(models)) 
        
    for i in range(len(models)):
        gmm    = models[i]  # Sprawdzanie modeli (każdy model wejściowy z każdym modelem w bazie)
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()

    detected = np.argmax(log_likelihood)
    print("\tDetected as - ", speakers[detected])
    time.sleep(1.0)  
        
#---------------------------------------------------------------------------------------------       
#Klasyfikator Neutralny

def test_model_neutral():
    print('Klasyfikator Neutral')
    os.chdir('/Users/Admin/Desktop/')
    source   = "../Desktop/train_speaker/testing_set_neutral/"  
    modelpath = "../Desktop/train_speaker/trained_models_neutral/"
    test_file = "../Desktop/train_speaker/testing_set_addition.txt"       
    file_paths = open(test_file,'r')
     
    gmm_files = [os.path.join(modelpath,fname) for fname in
                  os.listdir(modelpath) if fname.endswith('.gmm')]
     
    #Załadowanie modelów Gaussa
    models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
    speakers   = [fname.split("\\")[-1].split("*.gmm")[0] for fname 
                  in gmm_files]
     
    audio_path = '/Users/Admin/Desktop/BazaDanych-Ostatecznie/Emotional Speech Dataset (ESD) CREMA/1001/Angry/1001_DFA_ANG_XX.wav'

    sr,audio = read(audio_path)
    vector   = extract_MFCC(audio,sr)
            
    log_likelihood = np.zeros(len(models)) 
        
    for i in range(len(models)):
        gmm    = models[i]  # Sprawdzanie modeli (każdy model wejściowy z każdym modelem w bazie)
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()
 
    detected = np.argmax(log_likelihood)
    print("\tDetected as - ", speakers[detected])
    time.sleep(1.0)  

#---------------------------------------------------------------------------------------------     
#Klasyfikator Mieszany

def test_model_mix():
    print('Klasyfikator Mieszany')
    os.chdir('/Users/Admin/Desktop/')
    source   = "../Desktop/train_speaker/testing_set_mix/"  
    modelpath = "../Desktop/train_speaker/trained_models_mix/"
    test_file = "../Desktop/train_speaker/testing_set_addition.txt"       
    file_paths = open(test_file,'r')
     
    gmm_files = [os.path.join(modelpath,fname) for fname in
                  os.listdir(modelpath) if fname.endswith('.gmm')]
     
    #Załadowanie modelów Gaussa
    models    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
    speakers   = [fname.split("\\")[-1].split("*.gmm")[0] for fname 
                  in gmm_files]
     
    audio_path = '/Users/Admin/Desktop/BazaDanych-Ostatecznie/Emotional Speech Dataset (ESD) CREMA/1001/Angry/1001_DFA_ANG_XX.wav'

    sr,audio = read(audio_path)
    vector   = extract_MFCC(audio,sr)
            
    log_likelihood = np.zeros(len(models)) 
        
    for i in range(len(models)):
        gmm    = models[i]  # Sprawdzanie modeli (każdy model wejściowy z każdym modelem w bazie)
        scores = np.array(gmm.score(vector))
        log_likelihood[i] = scores.sum()

    detected = np.argmax(log_likelihood)
    print("\tDetected as - ", speakers[detected])
    time.sleep(1.0)  
        
#---------------------------------------------------------------------------------------------
#Wywołanie odpowiednich klasyfikatorów 

if y_pr_class_t == -1:
    test_model_mix()
if y_pr_class_t == 0:
    test_model_neutral()  
if y_pr_class_t == 1:
    test_model_happy()
if y_pr_class_t == 2:
    test_model_sad()
if y_pr_class_t == 3:
    test_model_angry() 
