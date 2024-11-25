# detect.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import cv2  # Import OpenCV for video processing
import numpy as np

# Load the saved model
model = load_model('image_captioning_model.keras')

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Determine the maximum sequence length (you need to define this function or load it)
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)

def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

# Map an integer to a word (you need to define this function)
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Generate a description for an image (same as in train.py)
def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_for_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    return in_text

def caption_image(image_path):
    # Load and preprocess the image
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    # Extract features
    feature = model.predict(image, verbose=0)

    # Generate caption
    caption = generate_desc(model, tokenizer, feature, max_length(descriptions))
    return caption

def caption_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame_count += 1
            if frame_count % 12 == 0:  # Caption every 12th frame
                # Save the frame as an image
                cv2.imwrite('temp_frame.jpg', frame)

                # Generate caption for the frame
                caption = caption_image('temp_frame.jpg')
                print(f"Frame {frame_count}: {caption}")

                # Display the resulting frame with caption (optional)
                cv2.putText(frame, caption, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Frame', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

# --- Main execution ---
if __name__ == "__main__":
    # Example usage for image captioning:
    image_path = 'path/to/your/image.jpg'
    caption = caption_image(image_path)
    print(caption)

    # Example usage for video captioning:
    video_path = 'path/to/your/video.mp4'
    caption_video(video_path)