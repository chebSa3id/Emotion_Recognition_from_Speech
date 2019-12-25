import glob
import os
import librosa
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.layers import Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[37]:


X=np.load('X.npy')
y=np.load('y.npy')



train_x, rest_x, train_y, rest_y = train_test_split(X, y, test_size=0.33,random_state=42)
test_x, val_x, test_y, val_y = train_test_split(rest_x,rest_y,test_size=0.5,random_state=42)



print(train_y)
print(rest_y)

print((test_y))
print((val_y))


# In[52]:


#dnn parameters
n_dim = train_x.shape[1]
n_classes = train_y.shape[1]
n_hidden_units_1 = n_dim
n_hidden_units_2 = 400 # approx n_dim * 2
n_hidden_units_3 = 200 # half of layer 2
n_hidden_units_4 = 100

#defining the model
def create_model(activation_function='relu', init_type='normal', optimiser='adam', dropout_rate=0.8):
    model = Sequential()
    # layer 1
    model.add(Dense(n_hidden_units_1, input_dim=n_dim, init=init_type, activation=activation_function))
    # layer 2
    model.add(Dense(n_hidden_units_2, init=init_type, activation=activation_function))
    model.add(Dropout(dropout_rate))
    # layer 3
    model.add(Dense(n_hidden_units_3, init=init_type, activation=activation_function))
    model.add(Dropout(dropout_rate))
    #layer4
    model.add(Dense(n_hidden_units_4, init=init_type, activation=activation_function))
    model.add(Dropout(dropout_rate))
    # output layer  n_classes
    model.add(Dense(n_classes, init=init_type, activation='sigmoid'))
    #model compilation
    model.compile(loss='binary_crossentropy', optimizer=optimiser, metrics=['accuracy'])
    return model


#create the model
model = create_model()
#train the model\\142
history = model.fit(train_x, train_y, validation_data=(val_x,val_y), epochs=200, batch_size=32)


'''
#model compilation
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#create the model
model = create_model()
#train the model\\142
history = model.fit(train_x, train_y, validation_data=(val_x,val_y), epochs=142, batch_size=4)
'''

# In[50]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[49]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[53]:


eva = model.evaluate(test_x,test_y)
print("Accuracy = ",eva[1]*100,"%  Error = ",eva[0]*100,'%')


# In[18]:


#predicting from the model
predict=model.predict(test_x,batch_size=32)


# In[19]:


emotions=['sad', 'happy']
#predicted emotions from the test set
y_pred=[]
for i in range(0,test_y.shape[0]):
    if predict[i][0]>0.5:
        y_pred.append(1)
    else :
        y_pred.append(0)
predicted_emo=[]
for i in range(0,test_y.shape[0]):
  emo=emotions[y_pred[i]]
  predicted_emo.append(emo)


# In[20]:


actual_emo=[]
y_true=test_y[:,0]
for i in range(0,test_y.shape[0]):
    emo=emotions[int(y_true[i])]
    actual_emo.append(emo)


# In[21]:
ii=0
cnt=0
for q in actual_emo :
    if actual_emo[ii]!=predicted_emo[ii]:
        cnt+=1
    print(ii+1,") Real : ",actual_emo[ii],"  Pred.: ",predicted_emo[ii])
    ii+=1

print("Correct = ",len(actual_emo)-cnt,"Wrong = ",cnt)


# In[ ]:

'''
model_name = 'Emotion_Voice_Detection_Model.h5'
save_dir = os.path.join(os.getcwd(), 'saved_models')
# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

import json
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

'''