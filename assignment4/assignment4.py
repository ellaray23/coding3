import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import collections


model_path = '/Users/ellaraypiddiu/Downloads/blaze_face_short_range.tflite'

#using mediapipe face detection
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,  #synchronous detection per frame
    min_detection_confidence=0.5
)
detector = vision.FaceDetector.create_from_options(options)



#define CNN class 
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

#load model
model = SimpleCNN(num_classes=7)
model.load_state_dict(torch.load("/Users/ellaraypiddiu/Downloads/simplecnn_fer2013.pth", map_location=torch.device('cpu')))
model.eval()

#define classes for FER2013
classes = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

#preprocessing transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

history = collections.deque(maxlen=5) #collects last 5 to get average class 

cap = cv2.VideoCapture(0) #for live camera


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect(mp_image)  #synchronous detection??

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

            #predict emotion
            with torch.no_grad():
                probs = torch.softmax(model(face_tensor), dim=1) #finds average emotion over last 5 for predicted class
                history.append(probs)
                avg_probs = torch.mean(torch.stack(list(history)), dim=0)
                pred_class = torch.argmax(avg_probs, dim=1).item()
                emotion = classes[pred_class]

            cv2.putText(frame, emotion, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Face + Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()


#still always on surprised