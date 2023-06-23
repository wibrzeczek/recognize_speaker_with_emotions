# recognize_speaker_with_emotions
My Python language project of a system based on a neural network and machine learning with TensorFlow and Keras frameworks used to recognize a speaker under the influence of emotions. Implemented tool has achieved the accuracy of recognizing speakers under at the level of 79.34%. Currently working on implementing the project to Alexa devices.

The source files included in the attachment are a representation of the created tool.
It consists of three programs:

- model_GMM_SR.py
The program is based on machine learning technology. Feature extraction functions for speaker recognition (MFCC, Chroma, Mel) were defined in it. The main functionalities of the file are – loading the training database, loading a set of test samples, training the model, testing the model. To run the program, you need to select the options in the order presented. The tool will load the training database first and then the test set. By means of appropriate conversions, it will change the names of samples to those adapted to the requirements of classifiers. Then, in the training function, the model performs feature extractions and saves Gaussian models for each of the three features for all database users. The last testing function allows you to make the appropriate GMM models for the test sample, and then recognize the speaker by comparing with the training models and determining the greatest similarity.


- model_LSTM_SER.py
The program is based on the LSTM classifier of neural networks. You must first define and attach the appropriate database and specify the names of all files saved during the training. These include: feature extraction files, model save files, save files of the best weights for the model. The program starts with the extraction of features for files (MFCC, ZCR, RMS), and then it creates appropriate lists to be divided into sets of training, testing and validation. The last step is to train the model and save it in the appropriate files on the disk. After the performed operations, the program presents the evaluation of the test and validation sets as well as prediction accuracy indicators.

- SER-tool.py
The program is a combination of the two above files. It is therefore responsible for recognizing speakers under the influence of emotions. To run it, it is necessary to run the previous two programs. An input sample is initially defined and will then be analyzed. The functions from the source file 'model_LSTM_SER.py' extract the features and call the model saved on the disk in order to verify the emotional tone contained in the audio file. Using the appropriate conditions and the value of the prediction variable, the sample goes to one of the 5 classifiers. At this point, the 'model_GMM_SR.py' program starts running. The sample is analyzed by one of the classifiers, and finally the answer is returned in the form of the speaker's index and the name of the voice feature thanks to which he was recognized.
