from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import imageDataGenerator 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense,Dropout, Flatten
from tensdorflow.keras.callbacks import EarlyStopping


train_path="data/train"
test_path="data/test"


train_data=imageDataGenerator(rescale=1./255,rotation_range=30,zoom_range=0.2,horizontal_flip=True)
test_data=imageDataGenerator(rescale=1./255),

