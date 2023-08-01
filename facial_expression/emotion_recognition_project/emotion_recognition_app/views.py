from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import tensorflow as tf
import numpy as np
import cv2
from mtcnn import MTCNN
import base64
import os

# Load the saved emotion recognition model
model_file_path = os.path.join(os.path.dirname(__file__), 'static/models/', 'finemodel.h5')
loaded_model = tf.keras.models.load_model(model_file_path)
IMG_SIZE = (224, 224)

# Function to preprocess the custom image
def preprocess_custom_image(face_img):
    img_array = np.expand_dims(face_img, axis=0)
    preprocessed_img = tf.keras.applications.vgg19.preprocess_input(img_array)
    return preprocessed_img

# Define class names containing the list of class names
class_names = ('angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')

def emotion_recognition_view(request):
    if request.method == 'POST' and request.FILES.get('custom_image'):
        # Get the uploaded file from the request
        custom_image = request.FILES['custom_image']

        # Check if the file was uploaded successfully
        if custom_image.content_type.startswith('image'):
            # Save the uploaded file to a temporary location
            fs = FileSystemStorage()
            filename = fs.save(custom_image.name, custom_image)

            # Process the uploaded image
            img_path = os.path.join(fs.location, filename)
            if os.path.isfile(img_path):
                img = cv2.imread(img_path)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Use MTCNN for face detection
                detector = MTCNN()
                faces = detector.detect_faces(rgb_img)

                if len(faces) > 0:
                    for i, face_info in enumerate(faces):
                        x, y, width, height = face_info['box']
                        x2, y2 = x + width, y + height

                        face_img = rgb_img[y:y2, x:x2]
                        face_img = cv2.resize(face_img, IMG_SIZE)

                        # Preprocess the face region
                        custom_image = preprocess_custom_image(face_img)

                        # Make predictions using the loaded model
                        predictions = loaded_model.predict(custom_image)
                        predictions = tf.nn.softmax(predictions)

                        # Get the predicted class index
                        predicted_class_index = tf.argmax(predictions, axis=1).numpy()[0]
                        predicted_class_label = class_names[predicted_class_index]

                        cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)  # Green bounding box

                        text = predicted_class_label
                        font_scale = 1.2
                        font_thickness = 2
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                        text_x = x + int((width - text_size[0]) / 2)
                        text_y = y + 25
                        cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

                    # Convert the image to Base64 format
                    _, buffer = cv2.imencode('.jpg', img)
                    base64_image = base64.b64encode(buffer).decode('utf-8')

                    # Delete the uploaded image file from the server
                    os.remove(img_path)

                    return render(request, 'emotion_recognition.html', {'faces_detected': True, 'detected_image': base64_image})

                # Delete the uploaded image file if no faces detected
                os.remove(img_path)

    return render(request, 'emotion_recognition.html', {'faces_detected': False})
