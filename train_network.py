"""
Multi-digit House Number Detection Using Convolutional Neural Networks
Author: John Hawkins
"""

from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, GlobalAveragePooling2D, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt


# unpack and preprocess data training/validation data (SVHN format 2)
X_train = loadmat('train_32x32.mat')['X'].transpose((3,0,1,2))
X_train = preprocess_input(X_train, mode='tf')
Y_train = loadmat('train_32x32.mat')['y'][:,0]
Y_train[Y_train == 10] = 0
Y_train = to_categorical(Y_train, num_classes=11)

# unpack training/validation negative images from CIFAR10. referenced from:
# https://towardsdatascience.com/cifar-10-image-classification-in-tensorflow-5b501f7dc77c
X_cifar_1 = loadmat('data_batch_1.mat')['data']
X_cifar_1 = X_cifar_1.reshape((len(X_cifar_1), 3, 32, 32)).transpose(0, 2, 3, 1)
Y_cifar_1 = np.ones((len(X_cifar_1))) * 10
Y_cifar_1 = to_categorical(Y_cifar_1, num_classes=11)
X_cifar_2 = loadmat('data_batch_2.mat')['data']
X_cifar_2 = X_cifar_2.reshape((len(X_cifar_2), 3, 32, 32)).transpose(0, 2, 3, 1)
Y_cifar_2 = np.ones((len(X_cifar_2))) * 10
Y_cifar_2 = to_categorical(Y_cifar_2, num_classes=11)
X_cifar_3 = loadmat('data_batch_3.mat')['data']
X_cifar_3 = X_cifar_3.reshape((len(X_cifar_3), 3, 32, 32)).transpose(0, 2, 3, 1)
Y_cifar_3 = np.ones((len(X_cifar_3))) * 10
Y_cifar_3 = to_categorical(Y_cifar_3, num_classes=11)
X_cifar_4 = loadmat('data_batch_4.mat')['data']
X_cifar_4 = X_cifar_4.reshape((len(X_cifar_4), 3, 32, 32)).transpose(0, 2, 3, 1)
Y_cifar_4 = np.ones((len(X_cifar_4))) * 10
Y_cifar_4 = to_categorical(Y_cifar_4, num_classes=11)
X_cifar_5 = loadmat('data_batch_5.mat')['data']
X_cifar_5 = X_cifar_5.reshape((len(X_cifar_5), 3, 32, 32)).transpose(0, 2, 3, 1)
Y_cifar_5 = np.ones((len(X_cifar_5))) * 10
Y_cifar_5 = to_categorical(Y_cifar_5, num_classes=11)

# unpack and preprocess data test data (SVHN format 2)
X_test = loadmat('test_32x32.mat')['X'].transpose((3,0,1,2))
X_test = preprocess_input(X_test, mode='tf')
Y_test = loadmat('test_32x32.mat')['y'][:,0]
Y_test[Y_test == 10] = 0
Y_test = to_categorical(Y_test, num_classes=11)

# unpack test negative images from CIFAR10
X_cifar_test = loadmat('test_batch.mat')['data']
X_cifar_test = X_cifar_test.reshape((len(X_cifar_test), 3, 32, 32)).transpose(0, 2, 3, 1)
Y_cifar_test = np.ones((len(X_cifar_test))) * 10
Y_cifar_test = to_categorical(Y_cifar_test, num_classes=11)

# concatenate SVHN and CIFAR data
X_train = np.concatenate((X_train, X_cifar_1, X_cifar_2, X_cifar_3, X_cifar_4, X_cifar_5), axis=0)
Y_train = np.concatenate((Y_train, Y_cifar_1, Y_cifar_2, Y_cifar_3, Y_cifar_4, Y_cifar_5), axis=0)
X_test= np.concatenate((X_test, X_cifar_test), axis=0)
Y_test = np.concatenate((Y_test, Y_cifar_test), axis=0)

# split train and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.10, random_state=5)

# augment images
datagen = ImageDataGenerator(rotation_range=20)
datagen.fit(X_train)

# initialize VGG16 pre trained model
# file_name = 'vgg_pretrained_weights_negclass'
# base_model = VGG16(weights='imagenet', include_top=False, input_shape=(32,32,3))
# for layer in base_model.layers[:-5]:
#     layer.trainable = False
# #x = GlobalAveragePooling2D()(base_model.output)
# x = Flatten(name='flatten')(base_model.output)
# x = Dense(4096, activation='relu')(x)
# x = Dropout(0.25)(x)
# x = Dense(4096, activation='relu')(x)
# x = Dropout(0.25)(x)
# x = Dense(11, activation='softmax', name='predictions')(x)
# model = Model(inputs=base_model.input, outputs=x)

# initialize VGG16 untrained model
# file_name = 'vgg_scratch_weights_negclass'
# input = Input(shape=(32,32,3), name='input')
# model = VGG16(weights=None, input_tensor=input, classes=11)

# initialize custom neural network model architecture
file_name = 'custom_weights_negclass'
model = Sequential()
model.add(Conv2D(32, kernel_size=3, input_shape=(32,32,3), activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2))
model.add(Dropout(0.25))
model.add(Conv2D(64, 3, activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2))
model.add(Dropout(0.25))
model.add(Conv2D(128, 3, activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2))
model.add(Dropout(0.25))
model.add(Conv2D(256, 3, activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2))
model.add(Dropout(0.25))
model.add(Conv2D(512, 3, activation='relu', padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(11, activation='softmax'))

print(model.summary())

# define training hyper parameters
if file_name == 'vgg_pretrained_weights_negclass':
    alpha = 0.0001
    epochs = 20
elif file_name == 'vgg_scratch_weights_negclass':
    alpha = 0.0001
    epochs = 20
else:
    alpha = 0.001
    epochs = 20
batch_size = 64

# compile model and train
optimizer = Adam(lr=alpha)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
callbacks = []
callbacks.append(ModelCheckpoint(filepath=file_name + '_best' + '.h5', monitor='loss', save_best_only=True))
callbacks.append(EarlyStopping(monitor= 'loss', min_delta=0.000001, patience=5, mode='auto'))
#callbacks.append(ReduceLROnPlateau(monitor = 'loss', factor = 0.1, verbose = 1, patience= 5, cooldown= 1, min_lr = 0.0001))
metrics = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_val, Y_val), callbacks=callbacks)
#metrics = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), epochs=epochs, verbose=1, validation_data=(X_val, Y_val), callbacks=callbacks)
score = model.evaluate(X_test, Y_test)
print('Test Loss:', score[0])
print('Test Accuracy:', score[1])
model.save(file_name + '_final' + '.h5')

plt.plot(metrics.history['acc'])
plt.plot(metrics.history['val_acc'])
plt.title(file_name + ' Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Training', 'Validation'], loc='upper left')
#plt.show()
plt.savefig(file_name + '_accuracy')
plt.close()

plt.plot(metrics.history['loss'])
plt.plot(metrics.history['val_loss'])
plt.title(file_name + ' Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'], loc='upper left')
#plt.show()
plt.savefig(file_name + '_loss')
