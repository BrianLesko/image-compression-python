# Brian Lesko 
# 1/21/2024
# Data science studies, real time live data display

import streamlit as st
import numpy as np
import customize_gui as gui
gui = gui.gui()
from sklearn.cluster import KMeans
import cv2 # !pip install opencv-python
import base64

def count_unique_colors(image):
    # Decode the image
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape to a list of colors
    img_list = img.reshape(-1, 3)

    # Count unique colors
    unique_colors = np.unique(img_list, axis=0)
    return unique_colors.shape[0]

def compress_image(image, num_clusters=0, jpeg_quality=75, crop_to_square=False, square_size=None, blur=0):
    # Decode the image
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Optional: Crop the image to a square
    if crop_to_square:
        h, w, _ = img.shape
        min_dim = min(h, w)
        
        # Crop to center square
        start_h = (h - min_dim) // 2
        start_w = (w - min_dim) // 2
        img = img[start_h:start_h + min_dim, start_w:start_w + min_dim]
        
        # Resize to the desired square size
        if square_size and square_size <= min_dim:
            img = cv2.resize(img, (square_size, square_size), interpolation=cv2.INTER_AREA)

    # Optional: Apply Gaussian blur
    if blur > 0:
        img = cv2.GaussianBlur(img, (blur, blur), 0)

    if num_clusters > 0:
        # Reshape for k-means
        img_list = img.reshape(-1, 3)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans.fit(img_list)

        # Map cluster centers back to the image
        compressed_img_list = kmeans.cluster_centers_[kmeans.labels_]
        compressed_img_array = compressed_img_list.reshape(img.shape).astype(np.uint8)
    else:
        # If num_clusters <= 0, use the original image
        compressed_img_array = img

    # Re-encode the image with JPEG quality settings
    _, encoded_image = cv2.imencode('.jpg', cv2.cvtColor(compressed_img_array, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return compressed_img_array, encoded_image

def main():
    gui.clean_format(wide=True)
    Sidebar = gui.about(text = "This code implements Kmeans clustering on an image")
    st.title("Image Compression")
    st.write("This is a simple example of Kmeans clustering on an image. The image is compressed to a specified number of colors.")
    
    col1, col2 = st.columns(2)
    with col1: before = st.empty()
    with col2: after = st.empty()

    with st.sidebar: 
        image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        num_colors = count_unique_colors(image) if image else 0
        num_clusters = st.slider("Number of colors", 0, num_colors, 25)
        jpeg_quality = st.slider("JPEG Quality", 1, 100, 75)
        crop_to_square = st.checkbox("Crop to square")
        square_size = st.slider("Square size", 100, 1000, 500) if crop_to_square else None
        blur = st.slider("Gaussian blur", 0, 10, 0)

    if image is not None:
        with col1: 
            st.image(image, use_container_width=True)
            st.write("Image uploaded successfully.")
            st.expander("Image Size", expanded=True).write(f"Image size: {image.size} bytes")
            st.write(f"Unique colors: {num_colors}")
            # make binary
            binary = image.read()
            st.expander(f"Binary Image Size: {len(binary)} bytes").write(binary)
            encoded = base64.b64encode(binary).decode()
            st.expander(f"Base64 Image Size: {len(encoded)} bytes").write(encoded)


        with st.spinner("Compressing image..."):
            compressed_img_array, encoded_image = compress_image(binary, num_clusters, jpeg_quality, crop_to_square, square_size, blur)
            compressed_size = len(encoded_image)  # Get compressed size

        with col2:
            st.image(compressed_img_array, use_container_width=True, caption="Compressed Image")
            # make binary from the compressed image, compressed_img_array
            binary_compressed_image = encoded_image.tobytes()
            st.expander(f"Binary Compressed Image Size: {len(binary_compressed_image)} bytes").write(binary_compressed_image)
            encoded_compressed = base64.b64encode(binary_compressed_image).decode()
            st.expander(f"Base64 Compressed Image Size: {len(encoded_compressed)} bytes").write(encoded_compressed)
            

main()