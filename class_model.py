
"""hand_signs"""

import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

import random

# Commented out IPython magic to ensure Python compatibility.

from zipfile import ZipFile

with ZipFile("/content/archive.zip","r")as zObject:
  zObject.extractall(path="/content/archive")
zObject.close()

train = pd.read_csv('/content/archive/sign_mnist_train.csv')
test = pd.read_csv('/content/archive/sign_mnist_test.csv')

label = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
print(len(label))

"""taking labels of attributes"""

train_label = train.label
test_label = test.label

train_data = np.array(train,dtype="float32")
test_data = np.array(test,dtype="float32")

i = random.randint(1,train.shape[0])
fig1,ax1 = plt.subplots(figsize=(2,2))
plt.imshow(train_data[i,1:].reshape((28,28)),cmap="gray")
print("label for image is ",label[train_label[i]])

fig = plt.figure(figsize=(18,18))
ax1 = fig.add_subplot(221)
train["label"].value_counts().plot(kind="bar",ax=ax1)
ax1.set_train_label("counts")
ax1.set_title("label")

x_train_data= train_data[:,1:]/255.0
x_test_data = test_data[:,1:]/255.0

y_train = train_data[:,0]
y_test = test_data[:,0]

x_train = x_train_data.reshape(-1,28,28,1)
x_test = x_test_data.reshape(-1,28,28,1)



"""model layers"""

model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(32,(3,3),input_shape=(28,28,1),activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (1,1)))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu'))
#model.add(tf.keras.layers.MaxPooling2D(pool_size =(2,2)))
model.add(tf.keras.layers.Dense(258,activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))


model.add(tf.keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))


model.add(tf.keras.layers.Conv2D(128,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(254,(3,3),activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(tf.keras.layers.Conv2D(254,(3,3),activation='relu'))



model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(256,activation='relu'))
model.add(tf.keras.layers.Dense(25,activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=["accuracy"])

model.summary()

history = model.fit(x_train,train_label, epochs = 10)



loss  = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss)  + 1)
plt.plot(epochs,loss ,'y',label='traning loss')
plt.plot(epochs,val_loss,'r',label='validation loss')
plt.title('training and validation loss')
plt.xlabel('Eposchs')
plt.ylabel('loss')
plt.show()



acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs , acc , 'y', label = 'trainin acc')
plt.plot(epochs, val_acc , 'r', label = 'Validation acc')
plt.title('training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

testM = test.iloc[:,1:].values
i = random.randint(1,7111)
print("actual label is ",label[test_label[i]])
a = testM[i]
print(a.shape)
a= np.array(a).reshape((- 1, 28, 28 ,1))
c = model.predict(a)
#print("predicted label is " ,label[np.argmax(model.predict)])
print(c)

m = np.where(c == 1)[1]
m = int(m)
print(m)
print("predicted value is ", label[m])

ab = np.array(a).reshape((28,28))
plt.imshow(ab)
plt.show()

model.save("SignLang.h5")
