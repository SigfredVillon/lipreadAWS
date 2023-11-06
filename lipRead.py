from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename  # Import secure_filename function
import os
import tensorflow as tf
from typing import List
import cv2
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
import ffmpeg
import dlib

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
    pwd = os.getcwd()
    hog_face_detector = dlib.get_frontal_face_detector()
    
    dlib_facelandmark = dlib.shape_predictor(pwd + "/data/shape_predictor_68_face_landmarks.dat")
    cap = cv2.VideoCapture(path)
    frames = []
    print(path)
    print("frames",int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = hog_face_detector(frame)

        #plt.imshow(gray2)
        face=faces[0]
        face_landmarks = dlib_facelandmark(gray, face)
                  
        # cv2.imshow('frames',frame)
        # start_point = ((face_landmarks.part(52).x) - 70, face_landmarks.part(52).y - 10)
        # end_point = ((face_landmarks.part(52).x) + 69, face_landmarks.part(52).y + 35)

        ytop=(face_landmarks.part(52).y)-10
        ybottom=(face_landmarks.part(52).y)+36
        xtop=(face_landmarks.part(52).x)-70
        xbottom=(face_landmarks.part(52).x)+70

        # print(f'frame[{ytop}:{ybottom},{xtop}:{xbottom},:]')
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[ytop:ybottom,xtop:xbottom,:])
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    # print(f'mean:{mean}, std:{std}')
    # print(f'frame-shape: {frames[0].shape}')
    cast = tf.cast((frames - mean), tf.float32) / std
    print(f'cast: {cast.shape}')
    return cast
    

def load_data(path: str): 
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]
    # File name splitting for windows
    #file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('vidFolder',f'{file_name}.mp4')
    frames = load_video(video_path) 

    
    return frames


def process_video(input_path, output_path, padding_image_path):
    try:
        # Open the input video
        cap = cv2.VideoCapture(input_path)

        # Define the desired frame count
        desired_frame_count = 75

        if cap.isOpened():
            # Get the total number of frames in the video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Load the padding image
            padding_frame = cv2.imread(padding_image_path)
            
            if padding_frame is None:
                print(f"Error: Failed to load the padding image from {padding_image_path}")
                return
            
            # Check if the padding image dimensions match the video frame dimensions
            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            if padding_frame.shape[0] != frame_height or padding_frame.shape[1] != frame_width:
                print("Error: Padding image dimensions do not match video frame dimensions.")
                return

            if total_frames == desired_frame_count:
                # If the video already has exactly 75 frames, just save it
                out = cv2.VideoWriter(output_path + "_part1.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
                for i in range(total_frames):
                    success, frame = cap.read()
                    if success:
                        out.write(frame)
                out.release()
                print(f"Saved exactly {desired_frame_count} frames to {output_path}_part1.mp4")
            elif total_frames < desired_frame_count:
                # If the video has less than 75 frames, add padding to reach 75 frames and save it
                out = cv2.VideoWriter(output_path + "_part1.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
                for i in range(total_frames):
                    success, frame = cap.read()
                    if success:
                        out.write(frame)

                for i in range(desired_frame_count - total_frames):
                    out.write(padding_frame)

                out.release()
                print(f"Added padding to reach {desired_frame_count} frames and saved to {output_path}_part1.mp4")
            else:
                # If the video has more than 75 frames, save the first 75 frames as part 1
                out = cv2.VideoWriter(output_path + "_part1.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
                for i in range(desired_frame_count):
                    success, frame = cap.read()
                    if success:
                        out.write(frame)
                out.release()
                print(f"Saved the first {desired_frame_count} frames to {output_path}_part1.mp4")

                # Create the second part
                out = cv2.VideoWriter(output_path + "_part2.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))
                for i in range(min(desired_frame_count, total_frames - desired_frame_count)):
                    success, frame = cap.read()
                    if success:
                        out.write(frame)

                for i in range(desired_frame_count - (total_frames - desired_frame_count)):
                    out.write(padding_frame)

                out.release()
                print(f"Saved the second part with {desired_frame_count} frames and added padding to {output_path}_part2.mp4")
        else:
            print("Failed to open the input video.")
    except Exception as e:
        print(f"Error: {str(e)}")





@app.route('/upload', methods=['POST'])
def upload_video():
    uploaded_file = request.files['video']
    if uploaded_file:
        padding_image_path = 'blank_frame.jpg'



        filename = secure_filename(uploaded_file.filename)
        

        changeName=filename.split('.')

        #saving of video
        video_path = os.path.join('vidFolder', changeName[0]+".mp4")
        uploaded_file.save(video_path)

       
        try:
           
           ffmpeg.input(video_path).output(os.path.join('vidFolder', changeName[0]+"edit"+".mp4"),vf='scale=360:288', r=25).run(overwrite_output=True)

        except ffmpeg.Error as e:
           print(f'Error during conversion: {e.stderr}')

        output_video_path = os.path.join('vidFolder', changeName[0]+".mp4")
        process_video(os.path.join('vidFolder', changeName[0]+"edit"+".mp4"), output_video_path, padding_image_path)
        

        combinedPrediction=""
        vidOne = os.path.join('vidFolder', changeName[0]+".mp4_part1.mp4")
        vidTwo = os.path.join('vidFolder', changeName[0]+".mp4_part2.mp4")


        vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
        char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
        # Mapping integers back to original characters
        num_to_char = tf.keras.layers.StringLookup(
        vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)

        model = load_model()
        if(os.path.isfile(vidOne)):

        # Process the video as needed
            print("VIDEO ONE IS ALIVE!")

            video= load_data(tf.convert_to_tensor(vidOne))

            yhat = model.predict(tf.expand_dims(video, axis=0))
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()


        # Convert prediction to text
            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
            combinedPrediction = combinedPrediction + " " + converted_prediction

        if(os.path.isfile(vidTwo)):

        # Process the video as needed
            print("VIDEO TWO IS ALIVE!")
            video= load_data(tf.convert_to_tensor(vidTwo))


            model = load_model()
            yhat = model.predict(tf.expand_dims(video, axis=0))
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()


        # Convert prediction to text
            converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
            combinedPrediction = combinedPrediction + " " + converted_prediction
        
  
          

        return jsonify({'Prediction': combinedPrediction})
    
    return jsonify({'error': 'No file uploaded'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
