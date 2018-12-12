import numpy as np
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Convolution2D
from keras.layers import Dense
from keras.layers import MaxPooling2D

classifier = Sequential()

classifier.add(Convolution2D(32,3,3,border_mode='same',input_shape=(64,64,3),activation='relu'))

classifier.add(MaxPooling2D(pool_size=(2,2)))

classifier.add(Flatten())

classifier.add(Dense(output_dim = 100,activation='relu'))
classifier.add(Dense(output_dim = 100,activation='relu'))
classifier.add(Dense(output_dim = 4,activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

data_generator_training = ImageDataGenerator(rescale=1/255,shear_range=0.4,horizontal_flip=True, zoom_range=0.56)
data_generator_test = ImageDataGenerator(rescale=1/255)

training_set = data_generator_training.flow_from_directory('dataset/training_set',target_size=(64,64),batch_size=20,class_mode='categorical')
test_set = data_generator_test.flow_from_directory('dataset/test_set',target_size=(64,64), batch_size=200, class_mode='categorical')

classifier.fit_generator(training_set,validation_data=test_set,epochs=10)






