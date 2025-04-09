# 1번
* MNIST 데이터셋을 이용하여 간단한 이미지 분류기를 만들어라.
## 전체 코드
```python
#MNIST 데이터셋 다운 및 로드

mnist = datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# 데이터 분할 확인
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

# 손글씨 숫자 이미지 크기 확인

image = train_x[0]
print(image.shape)

plt.imshow(image, 'gray')
plt.show()

# 데이터 정규화
from tensorflow.keras.utils import to_categorical

train_x = train_x.reshape(-1,28,28,1)/255.
test_x = test_x.reshape(-1,28,28,1)/255.
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

# 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,BatchNormalization,Conv2D

model = Sequential()
model.add(Conv2D(32,(2,2),activation="relu",input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(64,(2,2),activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(128,(2,2),2,activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(32,(2,2),activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(64,(2,2),activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(128,(2,2),2,activation="relu"))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(10,activation="softmax"))

# 모델 훈련
model.compile(loss = "categorical_crossentropy",optimizer = "adam",metrics=["acc"])
history = model.fit(train_x,train_y,validation_data=(test_x,test_y),epochs=10,batch_size=256)

# 훈련 결과
loss = history.history["loss"]
acc = history.history["acc"]
val_loss = history.history["val_loss"]
val_acc = history.history["val_acc"]
plt.subplot(1,2,1)
plt.plot(range(len(loss)),loss,label = "Train Loss")
plt.plot(range(len(val_loss)),val_loss,label = "Validation Loss")
plt.grid()
plt.legend()
plt.subplot(1,2,2)
plt.plot(range(len(acc)),acc,label = "Train Accuracy")
plt.plot(range(len(val_acc)),val_acc,label = "Validation Accuracy")
plt.grid()
plt.legend()
plt.show()

# 모델 평가
test_loss, test_accuracy = model.evaluate(test_x,test_y, verbose = 2)
print("Test Loss : ",test_loss)
print("Test Accuracy : ",test_accuracy)
```

## 작동 원리
1. 우선 MNIST 데이터셋을 다운받고 불러온다.
```python
mnist = datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()
```

2. MNIST 데이터가 훈련 세트와 테스트 세트와 분할되었는지 확인한다.
```python
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
```
![1번 0](https://github.com/user-attachments/assets/a9cb53c1-23fd-4705-862b-99de81cff1e8)

이와 같이 분할되었음을 알 수 있다.

3. 손글씨 숫자 이미지를 확인한다.
```python
image = train_x[0]
print(image.shape)

plt.imshow(image, 'gray')
plt.show()
```
![화면 캡처 2025-04-09 111704](https://github.com/user-attachments/assets/e2008a24-8be0-45cc-afd0-3d9ee26447e4)

4. 데이터를 0과 1 사이의 값으로 정규화한다.
```python
from tensorflow.keras.utils import to_categorical

train_x = train_x.reshape(-1,28,28,1)/255.
test_x = test_x.reshape(-1,28,28,1)/255.
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)
```

5. 이미지 분류기의 모델을 구성한다.
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,BatchNormalization,Conv2D

model = Sequential()
model.add(Conv2D(32,(2,2),activation="relu",input_shape=(28,28,1)))
model.add(BatchNormalization())
model.add(Conv2D(64,(2,2),activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(128,(2,2),2,activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(32,(2,2),activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(64,(2,2),activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(128,(2,2),2,activation="relu"))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(10,activation="softmax"))
```

6. 모델을 컴파일하고, 훈련한다.
```python
# 모델 훈련
model.compile(loss = "categorical_crossentropy",optimizer = "adam",metrics=["acc"])
history = model.fit(train_x,train_y,validation_data=(test_x,test_y),epochs=10,batch_size=256)

# 훈련 결과
loss = history.history["loss"]
acc = history.history["acc"]
val_loss = history.history["val_loss"]
val_acc = history.history["val_acc"]
plt.subplot(1,2,1)
plt.plot(range(len(loss)),loss,label = "Train Loss")
plt.plot(range(len(val_loss)),val_loss,label = "Validation Loss")
plt.grid()
plt.legend()
plt.subplot(1,2,2)
plt.plot(range(len(acc)),acc,label = "Train Accuracy")
plt.plot(range(len(val_acc)),val_acc,label = "Validation Accuracy")
plt.grid()
plt.legend()
plt.show()
```
훈련 결과는 다음과 같다.
![1번 결과 1](https://github.com/user-attachments/assets/1f5be94a-0476-4866-921d-68b2299255a7)
![1번 결과 2](https://github.com/user-attachments/assets/2f34044f-e78c-46fb-8ecc-ad830b154a73)


7. 모델을 테스트 세트를 통해 평가한다.
```python
test_loss, test_accuracy = model.evaluate(test_x,test_y, verbose = 2)
print("Test Loss : ",test_loss)
print("Test Accuracy : ",test_accuracy)
```
## 결과
![1번 결과 3](https://github.com/user-attachments/assets/a8bdaab8-dfca-4471-a8d1-7fb2c136ffae)


# 2번
* CIFAR10 데이터셋을 활용하여 합성곱 신경망(CNN) 구축하고, 이미지 분류를 수행하라.

## 전체 코드
```python
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
```

## 작동 원리
1. CIFAR 데이터를 불러온다.
```python
(train_x, train_y), (test_x, test_y) = cifar10.load_data()
```
2. 데이터 전처리를 수행한다.
```python
# 데이터 정규화
train_x = np.array(train_x/255.0, dtype = np.float32)
test_x = np.array(test_x/255.0, dtype = np.float32)

# 2차원인 labels 데이터를 1차원으로 변경.
train_y = train_y.squeeze()
test_y = test_y.squeeze()
```
3. 모델을 구성한다.
```python
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
```
4. 모델을 컴파일하고, 훈련한다.
```python
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
```
![2번 결과 1](https://github.com/user-attachments/assets/91c7fd3c-e93e-49d9-bfa3-e27f8edf0b7c)

5. 성능을 평가하고, 테스트 이미지에 대한 예측을 수행한다.
```python
# 테스트 결과
test_loss, test_acc = model.evaluate(test_x,  test_y, verbose=2)
print('Test loss : ',test_loss)
print('Test accuracy : ',test_acc)

# 데이터 예측
preds = model.predict(test_x)

# 예측 결과
print('예측 값 : ',preds[0].argmax())
plt.imshow(test_x[0])
plt.show()
print('정답 : ', preds[0].argmax())
```

## 결과
![2번 결과 2 5](https://github.com/user-attachments/assets/6c55a808-0740-402b-828e-2f1edc6266aa)

![2번 결과 3](https://github.com/user-attachments/assets/63880791-ab88-42f2-b0fa-7e8cbe15319e)
