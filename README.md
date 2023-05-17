## 在powershell里面输入：

```

Invoke-WebRequest -Uri "https://storage.googleapis.com/learning-datasets/rps.zip" -OutFile "D:\learning_datasets"


Invoke-WebRequest -Uri "https://storage.googleapis.com/learning-datasets/rps-test-set.zip" -OutFile "D:\learning_datasets"


```

得到如下数据集：

![image-20230516232653230](./assets/image-20230516232653230.png)

![image-20230516232700598](./assets/image-20230516232700598.png)

加载数据集：

```python
import os
import zipfile
local_zip = 'D:/learning_datasets/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('D:/learning_datasets/')
zip_ref.close()

local_zip = 'D:/learning_datasets/rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('D:/learning_datasets/')
zip_ref.close()
local_zip = 'D:/learning_datasets/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('D:/learning_datasets/')
zip_ref.close()

local_zip = 'D:/learning_datasets/rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('D:/learning_datasets/')
zip_ref.close()
```

```
rock_dir = os.path.join('D:/learning_datasets/rps/rock')
paper_dir = os.path.join('D:/learning_datasets/rps/paper')
scissors_dir = os.path.join('D:/learning_datasets/rps/scissors')

print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

rock_files = os.listdir(rock_dir)
print(rock_files[:10])

paper_files = os.listdir(paper_dir)
print(paper_files[:10])

scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10])

```

## 由于在本地训练模型的时候过慢，所以我尝试在colab运行，首先将本地下载好的数据集上传到google drive，然后在colab中挂载google drive

![image-20230517091628081](./assets/image-20230517091628081.png)

```
from google.colab import drive

drive.mount('/content/drive')
```

加载数据：

```
import os
import zipfile
local_zip = '/content/drive/MyDrive/learning_datasets/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content/drive/MyDrive/learning_datasets/')
zip_ref.close()

local_zip = '/content/drive/MyDrive/learning_datasets/rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content/drive/MyDrive/learning_datasets/')
zip_ref.close()
local_zip = '/content/drive/MyDrive/learning_datasets/rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content/drive/MyDrive/learning_datasets/')
zip_ref.close()

local_zip = '/content/drive/MyDrive/learning_datasets/rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/content/drive/MyDrive/learning_datasets/')
zip_ref.close()
```

训练模型：

```python
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR = "/content/drive/MyDrive/learning_datasets/rps/"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

VALIDATION_DIR = "/content/drive/MyDrive/learning_datasets/rps-test-set/"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)

model.save("rps.h5")
```





![image-20230517104614135](./assets/image-20230517104614135.png)

colab运行了一个多小时之后终于训练完成

![image-20230517112145954](./assets/image-20230517112145954.png)

完成模型训练之后，我们绘制训练和验证结果的相关信息。

```python
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()

```

![image-20230517112312298](./assets/image-20230517112312298.png)

