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
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
warnings.filterwarnings("ignore")

os.chdir('/Users/Admin/Desktop/')
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

# Załadowanie bazy danych dla trenowania modelu
def load_audio_train():
    os.chdir('/Users/Admin/Desktop/train_speaker/training_set_neutral')
    count=1
    
    for path, dirnames, filenames in os.walk('/Users/Admin/Desktop/train_speaker/training_set_neutral/'):
        print(path)
        print(dirnames)
        print(filenames)
        path1 = path.split('/')[-1]
        path1 = path.split("\\")[-1]
        if path1 != "Sad":
            continue
        else:
            for filename in filenames:
                os.chdir(path)
                FORMAT = pyaudio.paInt16
                CHANNELS = 1
                RATE = 44100
                CHUNK = 512
                RECORD_SECONDS = 3
                device_index = 2
                file1=filename
                name = filename.split('_')[0]
                file_out=name+"-sample"+str(count)+".wav"
                
                os.rename(file1,file_out)
                os.chdir('/Users/Admin/Desktop/train_speaker')
                file_trained_list = open("training_set_addition_s.txt", 'a')
                file_trained_list.write(file_out+"\n")
                os.chdir('/Users/Admin/Desktop/train_speaker/training_set_neutral/')

                count+=1

# Załadowanie bazy danych dla testowania modelu
def load_audio_test():
    os.chdir('/Users/Admin/Desktop/train_speaker/testing_set_neutral')
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 512
    RECORD_SECONDS = 10
    device_index = 2
    
    count=1
    for filename in os.listdir('/Users/Admin/Desktop/train_speaker/testing_set_neutral/'):

        file1=filename
        name = filename.split('_')[0]
        file_out=name+"-sample_train"+str(count)+".wav"

        os.rename(file1,file_out)
        os.chdir('/Users/Admin/Desktop/train_speaker')
        file_trained_list = open("testing_set_addition.txt", 'a')
        file_trained_list.write(file_out+"\n")
        os.chdir('/Users/Admin/Desktop/train_speaker/testing_set_neutral')

        count+=1

#--------------------------------------------------NAGRYWANIE PRÓBEK W CZASIE RZECZYWISTYM-----------------------------------

# Nagrywanie próbek do zestawy trenowania
def record_audio_train():
    
    os.chdir('/Users/Admin/Desktop/reco_mfcc_gmm')
    Name =(input("Please Enter Your Name:"))
    for count in range(5):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 512
        RECORD_SECONDS = 10
        device_index = 2
        audio = pyaudio.PyAudio()
        print("----------------------Record device list---------------------")
        info = audio.get_host_api_info_by_index(0)
        numdevices = info.get('deviceCount')
        for i in range(0, numdevices):
                if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                    print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))

        index = int(input())		
        print("Recording via index "+str(index))
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,input_device_index = index,
                        frames_per_buffer=CHUNK)
        print ("Recording started")
        Recordframes = []
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            Recordframes.append(data)
        print ("recording stopped")
        stream.stop_stream()
        stream.close()
        audio.terminate()
        file_out=Name+"-sample"+str(count)+".wav"
        WAVE_file_out=os.path.join("training_set",file_out)
        file_trained_list = open("training_set_addition.txt", 'a')
        file_trained_list.write(file_out+"\n")
        waveFile = wave.open(WAVE_file_out, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(Recordframes))
        waveFile.close()

# Nagrywanie próbek do zestawy testowania
def record_audio_test():
    os.chdir('/Users/Admin/Desktop/reco_mfcc_gmm')
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 512
    RECORD_SECONDS = 10
    device_index = 2
    audio = pyaudio.PyAudio()
    print("----------------------Record device list---------------------")
    info = audio.get_host_api_info_by_index(0)
    numdevices = info.get('deviceCount')
    for i in range(0, numdevices):
            if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                print("Input Device id ", i, " - ", audio.get_device_info_by_host_api_device_index(0, i).get('name'))
    print("-------------------------------------------------------------")
    index = int(input())		
    print("Recording via index "+str(index))
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,input_device_index = index,
                    frames_per_buffer=CHUNK)
    print ("Recording started")
    Recordframes = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        Recordframes.append(data)
    print ("Recording stopped")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    file_out="sample.wav"
    WAVE_file_out=os.path.join("testing_set",file_out)
    file_trained_list = open("testing_set_addition.txt", 'a')
    file_trained_list.write(file_out+"\n")
    waveFile = wave.open(WAVE_file_out, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(Recordframes))
    waveFile.close()

# Trenowanie modelu dla próbek rzeczywistych
def train_model_record():
    os.chdir('/Users/Admin/Desktop/')
    source = "../Desktop/reco_mfcc_gmm/training_set/"   
    dest = "../Desktop/reco_mfcc_gmm/trained_models/"
    train_file = "../Desktop/reco_mfcc_gmm/training_set_addition.txt"        
    file_paths = open(train_file,'r')

    count = 1
    features = np.asarray(())
    for path in file_paths:    
        path = path.strip()   
        sr,audio = read(source + path)
        vector   = extract_MFCC(audio,sr)
        
        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

        if count == 5:    
            gmm = GaussianMixture(n_components = 6, max_iter = 200, covariance_type='diag', n_init = 3)
            gmm.fit(features)
        
            file_picking = path.split("-")[0]+".gmm"
            pickle.dump(gmm,open(dest + file_picking,'wb'))
            print('+ modeling completed for speaker:',file_picking," with data point = ",features.shape)   
            features = np.asarray(())
            count = 0
        count = count + 1

# Testowanie modelu dla próbek rzeczywistych
def test_model_record():
    os.chdir('/Users/Admin/Desktop/')
    source   = "../Desktop/reco_mfcc_gmm/testing_set/"  
    modelpath = "../Desktop/reco_mfcc_gmm/trained_models/"
    test_file = "../Desktop/reco_mfcc_gmm/testing_set_addition.txt"       
    file_paths = open(test_file,'r')
     
    gmm_files = [os.path.join(modelpath,fname) for fname in
                  os.listdir(modelpath) if fname.endswith('.gmm')]
     
    # Załadowanie modelów Gaussa
    models_saved    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
    speakers   = [fname.split("\\")[-1].split(".gmm")[0] for fname 
                  in gmm_files]
     
    # Przeczytanie zestawy testowego
    # Otrzymanie listy testowych plików audio
    for path in file_paths:   
         
        path = path.strip()   
        print(path)
        sr,audio = read(source + path)
        vector   = extract_MFCC(audio,sr)
         
        log_likelihood = np.zeros(len(models_saved)) 
        
    # Sprawdzanie modeli (każdy model wejściowy z każdym modelem w bazie)
        for i in range(len(models_saved)):
            gmm    = models_saved[i]  
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.sum()
         
        detected = np.argmax(log_likelihood)
        print("\tDetected as - ", speakers[detected])
        time.sleep(1.0)  
        
# Trenowanie modelu dla zestawów z bazy danych
def train_model():
    os.chdir('/Users/Admin/Desktop/')
    source = "../Desktop/train_speaker/training_set/"   
    dest = "../Desktop/train_speaker/trained_models/"
    train_file = "../Desktop/train_speaker/training_set_addition.txt"        
    file_paths = open(train_file,'r')

    features = np.asarray(())
    features1 = np.asarray(())
    features2 = np.asarray(())

    for diiir in os.listdir(source):
        os.chdir(source+diiir)
        for filename in os.listdir():
            prefix = filename.split('.')[0]
            prefix = prefix.split('-')[0]
        take_file = []
        os.chdir('/Users/Admin/Desktop/')
        train_file = "../Desktop/train_speaker/training_set_addition.txt"        
        file_paths = open(train_file,'r')
        for path in file_paths:
            path1 = path.strip()  
            path_pref = path1.split('.')[0]
            path_pref = path_pref.split('-')[0] 
            if path_pref == prefix:
                take_file.append(path1)
        
        for take in take_file:
            os.chdir('/Users/Admin/Desktop/')
            os.chdir(source+diiir)
            pwd = os.getcwd()
            sr,audio = read(take)
            vector   = extract_MFCC(audio,sr)  
            if features.size == 0:
                features = vector
            else:
                features = np.vstack((features, vector))
            os.chdir('/Users/Admin/Desktop/')
            os.chdir(source+diiir)
            vector1  = extract_chroma(take)
            vector1 = vector1.reshape(-1,1)
            if features1.size == 0:
                features1 = vector
            else:
                features1 = np.vstack((features1, vector))
                       
            vector2  = extract_mel(take)
            vector2 = vector2.reshape(-1,1)
            if features2.size == 0:
                features2 = vector
            else:
                features2 = np.vstack((features2, vector))
                
        os.chdir('/Users/Admin/Desktop/')
        gmm = GaussianMixture(n_components = 6, max_iter = 200, covariance_type='diag',n_init = 3)
        gmm.fit(features)
                    
        # Zapisanie modeli Gaussa dla MFCC
        file_picking = take.split("-")[0]+"-mfcc.gmm"
        pickle.dump(gmm,open(dest + file_picking,'wb'))
        
        gmm1 = GaussianMixture(n_components = 1, max_iter = 200, covariance_type='diag',n_init = 3)
        gmm1.fit(features1)
                    
        # Zapisanie modeli Gaussa dla CHROMA
        file_picking1 = take.split("-")[0]+"-chroma.gmm"
        pickle.dump(gmm,open(dest + file_picking1,'wb'))
        
        
        gmm2 = GaussianMixture(n_components = 6, max_iter = 200, covariance_type='diag',n_init = 3)
        gmm2.fit(features2)
                    
        # Zapisanie modeli Gaussa dla MEL
        file_picking2 = take.split("-")[0]+"-mel.gmm"
        pickle.dump(gmm,open(dest + file_picking2,'wb'))
        
        
        print('+ modeling completed for speaker:',file_picking," with data point = ",features.shape)   
        
        features = np.asarray(())
        features1 = np.asarray(())
        features2 = np.asarray(())


#----------------------------------------------PRZYKŁAD TESTOWANIA MODELU DLA PRÓBEK NEUTRALNYCH------------------------------------
#----------------------------Testowanie odbyło się dla wszystkich emocji poprzez zmienianie odpowiednich destynacji folderów w modelpath-------------------
def test_model_neutral():
    os.chdir('/Users/Admin/Desktop/')
    source   = "../Desktop/train_speaker/testing_set_neutral/"  
    modelpath = "../Desktop/train_speaker/trained_models_neutral/"
    test_file = "../Desktop/train_speaker/testing_set_addition.txt"       
    file_paths = open(test_file,'r')
     
    gmm_files = [os.path.join(modelpath,fname) for fname in
                  os.listdir(modelpath) if fname.endswith('.gmm')]
     
    # Załadowanie modeli Gaussa
    models_saved    = [pickle.load(open(fname,'rb')) for fname in gmm_files]
    speakers   = [fname.split("\\")[-1].split(".gmm")[0] for fname 
                  in gmm_files]

     
    # Przeczytanie zestawy testowego
    # Otrzymanie listy testowych plików audio
    for path in file_paths:   
         
        path = path.strip()   
        print(path)
        sr,audio = read(source + path)
        vector   = extract_MFCC(audio,sr)
        log_likelihood = np.zeros(len(models_saved)) 
        
        # Sprawdzanie modeli (każdy model wejściowy z każdym modelem w bazie)
        for i in range(len(models_saved)):
            gmm    = models_saved[i]  
            scores = np.array(gmm.score(vector))
            log_likelihood[i] = scores.m()
         
        detected = np.argmax(log_likelihood)
        print("\tDetected as - ", speakers[detected])
        time.sleep(1.0)  
        

while True:
    choice=int(input("\n 1.Load audio for training \n 2.Load audio for testing \n 3.Record audio for training\n 4.Record audio for testing\n 5.Train record model\n 6.Test record model\n 7. Train model from database\n 8. Test model neutral"))
    if(choice==1):
        load_audio_train()
    elif(choice==2):
        load_audio_test()
    elif(choice==3):
        record_audio_train()
    elif(choice==4):
        record_audio_test()
    elif(choice==5):
        train_model_record()
    elif(choice==6):
        test_model_record()
    elif(choice==7):
        train_model()
    elif(choice==8):
        test_model_neutral()
    if(choice>8):
        exit()
    
