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

#------------------------------------DEFINICJA FUNKCJI CECH DLA PLIKÓW--------------------------------

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
    
    file_angry = ['DFA_ANG_XX' , 'IEO_ANG_HI' , 'IEO_ANG_LO' , 'IEO_ANG_MD' , 'IOM_ANG_XX' , 'ITH_ANG_XX' , 'ITS_ANG_XX' , 'IWL_ANG_XX' , 
                  'IWW_ANG_XX', 'MTI_ANG_XX' , 'TAI_ANG_XX' , 'TIE_ANG_XX' , 'TSI_ANG_XX', 'WSI_ANG_XX']
    
    file_happy = ['DFA_HAP_XX' , 'IEO_HAP_HI' , 'IEO_HAP_LO' , 'IEO_HAP_MD' , 'IOM_HAP_XX' , 'ITH_HAP_XX' , 'ITS_HAP_XX' , 'IWL_HAP_XX' , 
                  'IWW_HAP_XX', 'MTI_HAP_XX' , 'TAI_HAP_XX' , 'TIE_HAP_XX' , 'TSI_HAP_XX', 'WSI_HAP_XX']

    file_sad = ['DFA_SAD_XX' , 'IEO_SAD_HI' , 'IEO_SAD_LO' , 'IEO_SAD_MD' , 'IOM_SAD_XX' , 'ITH_SAD_XX' , 'ITS_SAD_XX' , 'IWL_SAD_XX' , 
                  'IWW_SAD_XX', 'MTI_SAD_XX' , 'TAI_SAD_XX' , 'TIE_SAD_XX' , 'TSI_SAD_XX', 'WSI_SAD_XX']
    
    file_neutral = ['DFA_SAD_XX' , 'IEO_SAD_HI' , 'IEO_SAD_LO' , 'IEO_SAD_MD' , 'IOM_SAD_XX' , 'ITH_SAD_XX' , 'ITS_SAD_XX' , 'IWL_SAD_XX' , 
                  'IWW_SAD_XX', 'MTI_SAD_XX' , 'TAI_SAD_XX' , 'TIE_SAD_XX' , 'TSI_SAD_XX', 'WSI_SAD_XX']

length_of_sample = []
folder_path = '/Users/Admin/Desktop/BazaDanych-Ostatecznie/Emotional Speech Dataset (ESD) CREMA/'

for subdir, dirs, files in os.walk(folder_path):
  for file in files: 
    x, sr = librosa.load(path = os.path.join(subdir,file), sr = None)
    xtrim, index = librosa.effects.trim(x, top_db=30)
     
    length_of_sample.append(len(xtrim))

print('Maksymalna długość próbki:', np.max(length_of_sample))    


#----------------------------------WSTĘPNE PRZETWARZANIE DLA WSZYSTKICH PLIKÓW--------------------------------------


# Definiowanie wymaganych zmiennych
total_length = 173056 
length_of_frame = 2048
hop_length = 512

# Definiowane list
list_rms = []
list_zcr = []
list_mfcc = []
list_emotions = []


folder_path = '/Users/Admin/Desktop/BazaDanych-Ostatecznie/Emotional Speech Dataset (ESD) CREMA/'


for subdir, dirs, files in os.walk(folder_path):
  print(subdir)
  for file in files: 

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
X = np.concatenate((feature_zcr, feature_rms, feature_mfccs), axis=2)

# Zmiana kształtu zmiennej „Y” na 2D.
Y = np.asarray(list_emotions).astype('int8')

Y = np.expand_dims(Y, axis=1)


# Zapisanie zdefiniowanych tablic X, Y jako listy do plików json.

os.chdir('/Users/Admin/Desktop/train/')

feature_x_data = X.tolist() 

with open("X_datanew_6.json", "w") as put_file:
    json.dump(feature_x_data, put_file)

feature_y_data = Y.tolist()

with open("Y_datanew_6.json", "w") as put_file:
    json.dump(feature_y_data, put_file)


# Ładowanie plików X, Y json z powrotem do list
# Przekonwertowanie list na np.arrays

with open("X_datanew_6.json", 'r') as list_file:
  X = json.load(list_file)
X = np.asarray(X, dtype = 'float32')


with open("Y_datanew_6.json", 'r') as list_file:
  Y = json.load(list_file)
Y = np.asarray(Y, dtype = 'int8')

# Podział na zestawy treningowe, walidacyjne i testowe.

x_train, x_tosplit, y_train, y_tosplit = train_test_split(X, Y, test_size = 0.225, random_state = 1)
x_val, x_test, y_val, y_test = train_test_split(x_tosplit, y_tosplit, test_size = 0.310, random_state = 1)

# Tworzenie 0neHot Vectors dla Y

y_train_class = tf.keras.utils.to_categorical(y_train-1, 4 , dtype = 'int8')
val_class_y = tf.keras.utils.to_categorical(y_val-1, 4 , dtype = 'int8')


# Wyświetlenie podziału zestawów

print(np.shape(x_train))
print(np.shape(y_train))
print(np.shape(x_test))
print(np.shape(y_test))
print(np.shape(y_train_class))
print(np.shape(x_val))
print(np.shape(y_val))

os.chdir('/Users/Admin/Desktop/train/')

x_test = x_test.tolist() 

with open("x_test_data_6.json", "w") as put_file:
    json.dump(x_test, put_file)

y_test = y_test.tolist() 

with open("y_test_data_6.json", "w") as put_file:
    json.dump(y_test, put_file)

#-------------------------------DEFINIOWANIE MODELU--------------------------------

# Definicja struktury i parametrów modelu
model = Sequential()
model.add(layers.LSTM(64, return_sequences = True, input_shape=(X.shape[1:3])))
model.add(layers.LSTM(64))
model.add(layers.Dense(4, activation = 'softmax'))
print(model.summary())

# Definiowanie batch_size - NWD dla rozmiaru wszystkich zestawów
batch_size = 1
    
# Definiowanie ścieżki dla najlepszych wag modelu
weight_path = h5py.File('best_weights_6.hdf5','a')
best_weight_path = '/Users/Admin/Desktop/train/best_weights_6.hdf5'
weight_path.close()
callsave = callbacks.ModelCheckpoint(best_weight_path, save_best_only=True, monitor='val_categorical_accuracy', verbose=2, mode='max')


# Zmniejszenie szybkości trenowania po 20 epokach bez widocznej poprawy.
rlrop = callbacks.ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.1, patience=20)
                             
# Definicja trenowania modelu
model.compile(loss='categorical_crossentropy', optimizer='RMSProp', metrics=['categorical_accuracy'])

history = model.fit(x_train, y_train_class, epochs=100, batch_size = batch_size, validation_data = (x_val, val_class_y), callbacks = [callsave, rlrop])
for key in history.history:
    print(key)
    
# Definicja najlepszych wag dla modelu
model.load_weights(best_weight_path)


#--------------------------------------OCENA MODELU (LOSS,ACCURACY)---------------------------------------

# Wizualizacja trendu strat 
plt.plot(history.history['loss'], label='Loss (training data)')
plt.plot(history.history['val_loss'], label='Loss (validation data)')
plt.title('Straty dla trenowania i wizualizacji')
plt.ylabel('Loss value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()

# Wizualizacja trendu dokładności kategorialnej 
plt.plot(history.history['categorical_accuracy'], label='Acc (training data)')
plt.plot(history.history['val_categorical_accuracy'], label='Acc (validation data)')
plt.title('Dokładność kategorialna modelu')
plt.ylabel('Acc %')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()


#--------------------------------------OCENA ZBIORU WALIDACYJNEGO------------------------------

# Obliczenie wyniku walidayjnego
loss,acc = model.evaluate(x_val, val_class_y, verbose=2)


val_class_y = np.argmax(val_class_y, axis=1)
predictions = model.predict(x_val)
pred_class_y = np.argmax(predictions, axis=1)
      
print(np.shape(val_class_y))
print(np.shape(pred_class_y))

# Zdefiniowanie macierzy nieporozumień dla zestawu walidacyjnego
cm=confusion_matrix(val_class_y, pred_class_y)


index = ['neutral','happy', 'sad', 'angry']  
columns = ['neutral','happy', 'sad', 'angry']  
 
cm_df = pd.DataFrame(cm,index,columns)                      
plt.figure(figsize=(15,8))
ax = plt.axes()

sns.heatmap(cm_df, ax = ax, cmap = 'PuBu', fmt="d", annot=True)
ax.set_ylabel('Prawdziwa emocja')
ax.set_xlabel('Przewidywania emocji')
plt.show()


# Obliczenie dokładności przewidywania dla zestawu walidacji

val_validation = cm.diagonal()
summary_row = np.sum(cm,axis=1)
acc = val_validation / summary_row

print('Wskaźniki dokładności przewidywania zestawu walidacji:')
for e in range(0, len(val_validation)):
    print(index[e],':', f"{(acc[e]):0.4f}")


#---------------------------------------ZAPISANIE I ZAŁADOWANIE MODELU----------------------------------------


# Zapisanie modelu
model_json = model.to_json()
path_to_model = '/Users/Admin/Desktop/Praca_inzynierska/SER_program/Models/model_6_caly.json'
path_to_weights = '/Users/Admin/Desktop/Praca_inzynierska/SER_program/Weights/model_6_caly.h5'


with open(path_to_model, "w") as json_file:
    json_file.write(model_json)
    
model.save_weights(path_to_weights)
print("Model został zapisany")


# Wczytanie modelu

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


# Ładowanie plików x_test, y_test json
# Konwersja plików do postaci np.arrays

x_test = load( 'x_test_data_6.json')
x_test = np.asarray(x_test).astype('float32')

y_test = load('y_test_data_6.json')
y_test = np.asarray(y_test).astype('int8')

test_y_class = tf.keras.utils.to_categorical(y_test-1, 4, dtype = 'int8')


#-------------------------------OCENA ZESTAWU TESTOWEGO----------------------------------

loss, acc = model.evaluate(x_test, test_y_class, verbose=2)

# Zdefiniowanie macierzy nieporozumień dla zestawu walidacyjnego

test_y_class = np.argmax(test_y_class, axis=1)
predictions = model.predict(x_test)
pred_class_y = np.argmax(predictions, axis=1)

cm=confusion_matrix(test_y_class, pred_class_y)

index = ['neutral', 'happy', 'sad', 'angry']  
columns = ['neutral', 'happy', 'sad', 'angry']  
 
cm_df = pd.DataFrame(cm,index,columns)                      
plt.figure(figsize=(12,8))
ax = plt.axes()

sns.heatmap(cm_df, ax = ax, cmap = 'BuGn', fmt="d", annot=True)
ax.set_ylabel('Prawdziwa emocja')
ax.set_xlabel('Przewidywana emocja')
plt.show()


# Obliczanie wskaźników dokładności przewidywań zestawu testowego

val_test = cm.diagonal()
summary_row = np.sum(cm,axis=1)
acc = val_test / summary_row

print('Współczynniki dokładności predykcji zestawu testowego:')
for e in range(0, len(val_test)):
    print(index[e],':', f"{(acc[e]):0.4f}")