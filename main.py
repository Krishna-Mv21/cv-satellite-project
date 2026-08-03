from tensorflow.keras.models import Sequential
from tensorflow.keras.preproceesing.image import ImageDataGenarator
from tensorflow.keras.layers import Conv2D,MaxpoolingD,Dense,Flatten
from tensorflow.keras.callbacks import EarlyStopping
train_path="dataset/train"
test_path="dataset/test"
train_data=ImageDataGenerator(rescale=1./255,rotation_range=30,zoom_range=0.2)
test_data=ImageDataGenerator(rescale=1./255)
train_generator=train_data.flow_from_directory(train_path,target_size=(244,244),batch_size=20,class_mode="categorical")
test_generator=test_data.flow_from_directory(test_path,target_size=(244,244),batch_size=20,class_mode="categorical")