# Brian Lesko 
# 1/21/2024
# Data science studies, real time live data display

import streamlit as st
import numpy as np
import customize_gui as gui
gui = gui.gui()
from sklearn.cluster import KMeans
import cv2 # !pip install opencv-python

def compress_image(image, num_clusters):
    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_list = img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(img_list)
    # Replace each pixel value with its nearby centroid
    compressed_img_list = kmeans.cluster_centers_[kmeans.labels_]
    compressed_img_array = compressed_img_list.reshape(img.shape)
    # Normalize the pixel values to 0-1
    compressed_img_array = compressed_img_array / 255.0
    return compressed_img_array

def main():
    gui.clean_format(wide=True)
    Sidebar = gui.about(text = "This code implements Kmeans clustering on an image")
    st.title("Image Compression")
    st.write("This is a simple example of Kmeans clustering on an image. The image is compressed to a specified number of colors.")
    
    col1, col2 = st.columns(2)
    with col1: before = st.empty()
    with col2: after = st.empty()

    with st.sidebar: 
        image = st.file_uploader("Choose an image...", type="jpg")
    if image is not None:
        with col1: 
            st.image(image, use_column_width=True)
            st.write("Image uploaded successfully.")

        with st.spinner("Compressing image..."):
            compressed_image = compress_image(image, 16)  # Compress image to 16 colors
        with col2:
            st.image(compressed_image, use_column_width=True)
            st.write("Image compressed successfully.")

main() 