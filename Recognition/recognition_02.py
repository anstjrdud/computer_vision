# -*- coding: utf-8 -*-
"""Recognition_02.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1G3CP8UxjqcN3xTm-j7j9OpknXDa3vddL
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10

# 데이터 로드
(train_x, train_y), (test_x, test_y) = cifar10.load_data()

# 데이터 분할 확인
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

# 데이터 정규화
train_x = np.array(train_x/255.0, dtype = np.float32)
test_x = np.array(test_x/255.0, dtype = np.float32)

# 2차원인 labels 데이터를 1차원으로 변경.
train_y = train_y.squeeze()
test_y = test_y.squeeze()

# 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam

model = keras.Sequential()
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2)))
model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(10))

# 모델 훈련
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_x, train_y, epochs=10,
                    validation_data=(test_x, test_y))

# 훈련 결과
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

# 테스트 결과
test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=2)
print('Test loss : ',test_loss)
print('Test accuracy : ',test_acc)

# 데이터 예측
preds_data = test_x[20:30]
preds_label = test_y[20:30]
preds = model.predict(preds_data)

# 예측 결과
print(preds_label, end = ' ')
print()
for i in range(0, 10):
  print(np.argmax(preds[i]), end = ' ')
