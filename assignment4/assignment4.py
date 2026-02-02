import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import collections

#loading pre-trained MEDIAPIPE BLAZEFACE MODEL for face detection
model_path = '/Users/ellaraypiddiu/Downloads/blaze_face_short_range.tflite'

#https://colab.research.google.com/github/googlesamples/mediapipe/blob/main/examples/face_detector/python/face_detector.ipynb (as reference)
#setup for mediapipe face detection:
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,  #setting it to image mode so can detect ...
    min_detection_confidence=0.5 #setting the mminimum confidence to 0.5 so doesnt show below
)
detector = vision.FaceDetector.create_from_options(options)



#define CNN class (from assignment3)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

#loading the best version of my model: 
model = SimpleCNN(num_classes=7) #calling CNN class and defining amount of classes
model.load_state_dict(torch.load("/Users/ellaraypiddiu/Downloads/simplecnn_fer2013.pth", map_location=torch.device('cpu')))
model.eval()

classes = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

#transforms faces to greyscale and 48,48, converts to tensor and normalises 
#to match how model was trained
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

#for my robotic application I'm going to use the the smoothing as a frame 
#smoothing with deque - reducing noise and calculating average of last 5 predictions
history = collections.deque(maxlen=5) #i think actually creates more bias towards surprise
#bc it has such a high bias towards it, if i collect the last 5, it's more likely that it iwll be surprise, like if 3 of them are ..

frame_count = 0 #starting at 0 frames
max_frames = 50 #added 50 frames to represent the 20 second photo booth user interaction
frames_data = [] #adding a list to append all the data from each frame for drawing later

#adding a representation of the type of oval it would create
#would be connected to the drawing machine and change motor output, changing x, y axis to draw the correct shapes
#so shapes would be specific motor movements that I would add here
shape_mapping = { 
    "Happy": "big slightly spikey oval",
    "Neutral": "medium sized circle",
    "Angry": "small spikey oval",
    "Sad": "big round oval",
    "Surprise": "big spikey oval",
    "Fear": "medium sized spikey oval",
    "Disgust": "small distorted oval"
}

cap = cv2.VideoCapture(0) #for live camera


while cap.isOpened() and frame_count < max_frames: #so keeps the loop open (closes after) until it reaches 50 frames
    ret, frame = cap.read()
    if not ret:
        break

    #setup for mediapipe face detection:
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect(mp_image) #finds faces in current frame

    #bounding boxes for detections
    if result.detections:
        for detection in result.detections:
            box = detection.bounding_box
            box = detection.bounding_box
            x1, y1 = max(int(box.origin_x),0), max(int(box.origin_y),0)
            x2, y2 = min(int(box.origin_x + box.width), frame.shape[1]), min(int(box.origin_y + box.height), frame.shape[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            #crop face and convert to PIL
            face_img = frame[y1:y2, x1:x2]
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            face_tensor = transform(face_pil).unsqueeze(0)  #add batch dim

            #to predict emotion
            with torch.no_grad():
                probs = torch.softmax(model(face_tensor), dim=1) #finds average emotion over last 5 it detects 
                history.append(probs) #and appends to history to use after to find average class in frame:

            if len(history) == 5: #calculates the average from the last 5 predictions
                #so each deque average, history is a frame now
                avg_probs = torch.mean(torch.stack(list(history)), dim=0) #mean from history (deque smoothing)
                pred_class = torch.argmax(avg_probs, dim=1).item() #finds the max for class
                confidence = avg_probs[0, pred_class].item() #finds confidence of the predicted max class
                emotion = classes[pred_class] #becomes the emotion in that one frame
                
                #in real robotic application this would be line_type > 70 and confidence would be mapped
                #to the line shakiness so it would be line = map(confidence, 0, 10, 0, 100) in C++ for motor movement
                if confidence > 0.7:
                    line_style = "smooth line" #if the confidence score is high make it smooth line
                elif confidence > 0.4:
                    line_style = "moderately wobbly line" #if it's unsure but moderately okay its moderately wobbly
                else:
                    line_style = "completely wobbly line" #if it's completely unsure it's completely wobbly
                #this becomes a reflection of the model, the machine learning the user's emotion interaction

                #add 1 to the averaged frame count so it knows when it gets to 50
                frame_count += 1

                #save all the data to frames_data list 
                frames_data.append({
                    "frame": frame_count,
                    "emotion": emotion,
                    "confidence": round(confidence,2),
                    "shape": shape_mapping[emotion],
                    "line_style": line_style
                })

                #print what would be the robotic output / be sent to motors for drawing
                print(f"Frame {frame_count}: Emotion Detected: {emotion}, Confidence: {confidence:.2f}, Shape Drawn: {shape_mapping[emotion]}, Line Style: {line_style}")

                #clear history for next frame to repeat process 50x
                history.clear()

                #draws on live feed what emotion and confidence it thinks it's detecting
                cv2.putText(frame, f"{emotion} ({confidence:.2f})", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Face + Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

print("\nDrawing According to Emotions...")
for f in frames_data:
    print(f) #prints all the frames as articualted rather than drawing for now!


#problem with model:
#still always on surprised

#could be because dataset has more surprise 
#dataset has already proved to be wrong - fear predicted as anger when it definitely is anger (predictions more correct than true)
#the size of the dataset images are so small, could easily be distorted in live camera
#the lighting of webcam can confuse the model - CNN focus on edges and contrast and surprise has lots of contrast
#because of open mouth and eyes, where lighting in webcam can exaggerate it so always thinks people are surprised.
#trained on no background, the bounding boxes do include neck, hair, background so could be a factor.
#my averaging of the last 5 detected emotions could lead to strengthening the bias.

#though reflects how a machine can be wrong.