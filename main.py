import os
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical


train_data = pd.read_csv("Train.csv", encoding="utf-8")
test_data = pd.read_csv("Test.csv", encoding="utf-8")
meta_data = pd.read_csv("Meta.csv", encoding="utf-8")


num_classes = len(meta_data["ClassId"].unique())


def load_images(data, base_path, use_class_subfolders=True, img_size=(32, 32)):
    images, labels = [], []
    for _, row in data.iterrows():
        
        if use_class_subfolders:
            
            img_path = os.path.join(base_path, str(row["ClassId"]), os.path.basename(row["Path"]))
        else:
            
            img_path = os.path.join(base_path, os.path.basename(row["Path"]))

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size) 
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
            images.append(img)
            labels.append(row["ClassId"])
        else:
            print(f"⚠️ تحذير: الصورة {img_path} غير موجودة!")
    
    return np.array(images) / 255.0, np.array(labels)  


train_path = "C:/Users/Adel/Desktop/progect1/archive/Train"
test_path = "C:/Users/Adel/Desktop/progect1/archive/Test"


train_images, train_labels = load_images(train_data, train_path, use_class_subfolders=True)
test_images, test_labels = load_images(test_data, test_path, use_class_subfolders=False)


print(f"✅ عدد صور التدريب: {len(train_images)} | ✅ عدد صور الاختبار: {len(test_images)}")


train_labels_onehot = to_categorical(train_labels, num_classes)
test_labels_onehot = to_categorical(test_labels, num_classes)


model = Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),  
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(train_images, train_labels_onehot, epochs=10, batch_size=32, validation_data=(test_images, test_labels_onehot))


test_loss, test_acc = model.evaluate(test_images, test_labels_onehot)
print(f"✅ دقة النموذج على بيانات الاختبار: {test_acc:.4f}")


model.save("traffic_signs_model.h5")
print("📁 تم حفظ النموذج بنجاح!")
