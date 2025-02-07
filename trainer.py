import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern

#Load images and preprocess
def load_images_with_id(path):
    images_path = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    for single_image_path in images_path:
        faceImg = cv2.imread(single_image_path, cv2.IMREAD_GRAYSCALE)
        id = int(os.path.split(single_image_path)[-1].split(".")[1])
        faces.append(faceImg)
        ids.append(id)
    return np.array(ids), faces

    #Compute LBP features and histograms
def compute_lbp_and_histograms(faces):
    lbph_values = []
    lbp_histograms = []
    for face in faces:
        lbp = local_binary_pattern(face, 8, 1, 'default')
        lbp_resized = cv2.resize(lbp, (200, 200))
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        lbph_values.append(lbp_resized)
        lbp_histograms.append(hist)
    lbph_values = np.array(lbph_values)
    lbp_histograms = np.array(lbp_histograms)
    return lbph_values, lbp_histograms

#Train LBPH recognizer
def train_lbph_recognizer(faces, ids):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, ids)
    return recognizer

#Load images and preprocess
path = "cropped_face"
ids, faces = load_images_with_id(path)
#compute LBP features and histograms
lbph_values, lbp_histograms = compute_lbp_and_histograms(faces)
#Train LBPH recognizer
recognizer = train_lbph_recognizer(lbph_values, ids)

#Display LBP images and save them
for i, lbp_image in enumerate(lbph_values):
    plt.imshow(lbp_image, 'gray')
    plt.title(f'LBP Image for ID {ids[i]}')
    plt.axis('off')
    plt.savefig(f'lbp_image_id_{ids[i]}.jpg')
    plt.close()

#Display histograms and save them
for i, hist in enumerate(lbp_histograms):
    plt.plot(hist)
    plt.title(f'LBPH Histogram for ID {ids[i]}')
    plt.xlabel('Bins')
    plt.ylabel('Frequency')
    plt.savefig(f'histogram_id_{ids[i]}.jpg')
    plt.close()

#Save trained recognizer
recognizer.save("recognizer/trainingdata.yml")