from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename  # Import secure_filename function
import os
import tensorflow as tf
from typing import List
import cv2
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
import ffmpeg


app = Flask(__name__)


def load_model() -> Sequential: 
    model = Sequential()

    model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

    model.load_weights('models/checkpoint')

    return model

def load_video(path:str) -> List[float]: 
    #print(path)
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236,80:220,:])
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std
    

def load_data(path: str): 
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]
    # File name splitting for windows
    #file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('vidFolder',f'{file_name}.mpg')
    frames = load_video(video_path) 

    
    return frames



@app.route('/upload', methods=['POST'])
def upload_video():
    uploaded_file = request.files['video']
    if uploaded_file:

        #saving of video
        filename = secure_filename(uploaded_file.filename)



        filename.split('.')

        print("This is the filename",filename)
        try:
           
           ffmpeg.input(filename).output(filename+".mpg").run()
           
           print(f'Conversion completed: {filename+".mpg"}')

        except ffmpeg.Error as e:
           print(f'Error during conversion: {e.stderr}')


        video_path = os.path.join('vidFolder', filename+".mpg")
        uploaded_file.save(video_path)

        # Process the video as needed

        vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
        char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
        # Mapping integers back to original characters
        num_to_char = tf.keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)


        video= load_data(tf.convert_to_tensor(video_path))


        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()


        # Convert prediction to text
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')


        return jsonify({'Prediction': converted_prediction})
    
    return jsonify({'error': 'No file uploaded'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
