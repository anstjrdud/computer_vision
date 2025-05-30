# -*- coding: utf-8 -*-
"""Recognition_03.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cm9PnywahJ_hxHlUuAkY51ups__Yg_Wv
"""

from tensorflow.keras.applications import VGG16
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf

# 1. 데이터셋 불러오기
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 2. 이미지 전처리 (VGG16 전용)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. VGG16 모델 불러오기 (include_top=False)
base_model = VGG16(weights='imagenet', include_top = False, input_shape=(32, 32, 3))

# 4. 기존 가중치 고정 (freeze)
for layer in base_model.layers:
    layer.trainable = False

# 5. 새 레이어 추가
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dense(256, activation='relu')(x)
output = Dense(10, activation='softmax')(x)  # CIFAR-10 → 클래스 10개
model = Model(inputs=base_model.input, outputs=output)

# 6. 모델 컴파일
model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                 metrics=['accuracy'])

# 7. 모델 학습
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

# 8. MyModel 설계
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow import keras

my_model = Sequential([
  keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
  keras.layers.MaxPooling2D((2, 2)),
  keras.layers.Conv2D(128, (3, 3), activation='relu'),
  keras.layers.MaxPooling2D((2, 2)),
  keras.layers.Conv2D(128, (3, 3), activation='relu'),
  keras.layers.Flatten(),
  keras.layers.Dense(64, activation='relu'),
  keras.layers.Dense(32, activation='relu'),
  keras.layers.Dense(10, activation='softmax')

])

my_model.compile(optimizer='adam',
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                 metrics=['accuracy'])
my_model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

# 9. VGG16 모델과 MyModel 모델 성능 비교

loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
loss, accuracy = my_model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
