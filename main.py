import tensorflow as tf
import streamlit as st
import cv2
import os
from detect_faces import FaceDetector

model = tf.keras.models.load_model("models/model")

fd = FaceDetector()

st.set_page_config(page_title="Mask Detection Using Tensorflow", page_icon=None, layout='centered',
                   initial_sidebar_state='auto')
st.title("Mask Detection Using Tensorflow")


def main():
    image = st.file_uploader(label="Upload an Image with your face", type=['png', 'jpg', 'jpeg'])
    button = st.button("Upload Image")
    if button and image:
        save_uploaded_file(image)
        img_dir = "tempDir/" + image.name
        img = cv2.imread(img_dir)
        face_boxes, faces_score = fd.get_faceboxes(img, 0.5)
        if len(face_boxes) == 0:
            text = "Unfortunately No Face was Detected"
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            for facebox in face_boxes:
                face_img = img[facebox[1]: facebox[3], facebox[0]: facebox[2]]
                img = cv2.rectangle(img, (facebox[0], facebox[1]), (facebox[2], facebox[3]), (0, 255, 0))
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                results = predict(face_img).flatten()
                results = tf.nn.sigmoid(results)
                results = tf.where(results < 0.5, 0, 1)

                if results[0] == 0:
                    text = 'Mask Detected'
                else:
                    text = 'No mask'

        st.text(text)
        os.remove(img_dir)
    pass


def save_uploaded_file(uploaded_file):
    """

    :param uploaded_file:
    :return:
    """
    with open(os.path.join("tempDir", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    return None


def predict(img):
    img = pre_process(img)
    result = model.predict(img)

    return result


def pre_process(image):
    image = cv2.resize(image, (160, 160))
    image = image.reshape(1, *image.shape)

    return image


if __name__ == "__main__":
    main()
