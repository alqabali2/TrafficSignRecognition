import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import os


model = tf.keras.models.load_model("traffic_signs_model.h5")



test_data = pd.read_csv("Test.csv", encoding="utf-8")
test_path = "C:/Users/Adel/Desktop/progect1/archive/Test"


def load_test_images(data, base_path, img_size=(32, 32)):
    images, labels, paths = [], [], []
    for _, row in data.iterrows():
        img_path = os.path.join(base_path, os.path.basename(row["Path"]))

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img / 255.0)  # 
            labels.append(row["ClassId"])
            paths.append(img_path)
        else:
            print(f"⚠️ تحذير: الصورة {img_path} غير موجودة!")

    return np.array(images), np.array(labels), paths


test_images, test_labels, test_paths = load_test_images(test_data, test_path)


num_samples = 5  
random_indices = np.random.choice(len(test_images), num_samples, replace=False)
random_images = test_images[random_indices]
random_labels = test_labels[random_indices]


predictions = model.predict(random_images)
predicted_classes = np.argmax(predictions, axis=1)


plt.figure(figsize=(12, 6))
for i in range(num_samples):
    plt.subplot(1, num_samples, i + 1)
    plt.imshow(random_images[i])
    plt.title(f"الحقيقي: {random_labels[i]}\nالمتوقع: {predicted_classes[i]}")
    plt.axis("off")
plt.show()

print("✅ تم اختبار صور عشوائية جديدة!")
