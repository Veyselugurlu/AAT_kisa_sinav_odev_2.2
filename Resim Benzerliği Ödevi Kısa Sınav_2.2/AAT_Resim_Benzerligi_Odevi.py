# -*- coding: utf-8 -*-

import cv2
import os

# Resimlerin bulunduğu linkten bilgisayara indirilmiş olan klasörü belirtin
image_folder = "C:/Users/Veyse/Desktop/Resimler"

# Klasördeki resim dosyalarını alın. (.jpg ve .png) resim dosyaları için geçerlidir
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

# Benzerlik eşiği %90 üzeri olan resimleri yazdıracak 
threshold = 0.9

# SIFT özellik çıkarıcısı oluştur
sift = cv2.xfeatures2d.SIFT_create()

# Benzer resimleri tutmak için bir liste oluşturun
similar_images = []

# Resimleri yükle ve benzerlik kontrolünü yap
for i in range(len(image_files)):
    for j in range(i + 1, len(image_files)):
        # Resimleri yükle
        img1 = cv2.imread(os.path.join(image_folder, image_files[i]), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(os.path.join(image_folder, image_files[j]), cv2.IMREAD_GRAYSCALE)

        # Keypoint'leri ve açıklıkları bul
        kp1, desc1 = sift.detectAndCompute(img1, None)
        kp2, desc2 = sift.detectAndCompute(img2, None)

        # BFMatcher ile eşleşmeleri bul
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc1, desc2, k=2)

        # İyi eşleşmeleri seç
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        # Benzerlik puanını kontrol et ve eşik değerini aşanları listeye ekle
        similarity_ratio = len(good_matches) / len(matches)
        if similarity_ratio > threshold:
            similar_images.append((image_files[i], image_files[j], similarity_ratio))

# Benzer resimleri ve benzerlik oranlarını ekrana yazdır
if len(similar_images) > 0:
    print("Benzer Resimler:")
    for img_pair in similar_images:
        print(f"{img_pair[0]} ve {img_pair[1]} - Benzerlik Oranı: {img_pair[2]:.2f}")
# hiç resim bulunamazsa ekrana benzer resim bulunamadı yazdır
else:
    print("Benzer resim bulunamadı.")



