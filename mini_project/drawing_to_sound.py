import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import random
import os
import pygame 


#define CNN class 
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) #1 input channel - GREYSCALE
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 64 * 64, 128)  #dataset images are 256x256; after pooling twice size is 64x64
        self.fc2 = nn.Linear(128, num_classes) #number of classes: 4 (right now)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # Output size: [batch,32,128,128]
        x = self.pool(F.relu(self.conv2(x)))   # Output size: [batch,64,64,64]
        x = x.view(-1, 64 * 64 * 64)             # Flatten feature maps to vector
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)                        # No softmax needed with CrossEntropyLoss
        return x

#loading my spirograph detection model
model = SimpleCNN(num_classes=4) #calling CNN class and defining amount of classes
model.load_state_dict(torch.load("/Users/ellaraypiddiu/Downloads/spiro_miniproject.pth", map_location=torch.device('cpu')))
model.eval()

#defining classes for detection
classes = ['high_loud','high_quiet','low_loud','low_quiet'] 

#transforming frames of video spirograph data - prepares frames to feed into CNN
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


#for playing sounds in python 
pygame.mixer.init() #initialises the mixer (audio system)

#get computer sounds to apply to detected spirograph classes
sound_base_path = "/Users/ellaraypiddiu/Downloads/spyrograph-main/mini_project/sound_samples"
sound_files = {}

for soundclass in classes: #finding correlating class - sound path 
    folder = os.path.join(sound_base_path, soundclass)
    sound_files[soundclass] = [pygame.mixer.Sound(os.path.join(folder, f)) #using mixer.Sound for sound playback
                         for f in os.listdir(folder) if f.endswith('.wav')] #skips corrupted sounds

current_soundclass = None #using to cut off the current sound class and go to the next one for next detected spirograph 

#background subtraction settings
backSub = cv2.createBackgroundSubtractorMOG2(
    history=200, #doesnt make a difference because they arent moving and switch quickly
    varThreshold=1, #love the effect
    detectShadows=True
)

cap = cv2.VideoCapture("/Users/ellaraypiddiu/Downloads/Untitled_Artwork3.MP4") #to upload video

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    #preprocessing frame so it can go into my cnn - needs video frames to be pil images
    pil_image = Image.fromarray(frame)
    input_tensor = transform(pil_image).unsqueeze(0) #applying tranforms, unsqueeze adds batch dimension

    #my model prediction:
    with torch.no_grad():
        outputs = model(input_tensor)
        pred_idx = torch.argmax(outputs, dim=1).item()
        pred_class = classes[pred_idx]

    #choosing a random sound from the correlating predicted class 
    #(sound_samples are in high_loud etc. classes too)
    if sound_files.get(pred_class):
        #stop previous sound
        if current_soundclass != None:
            current_soundclass.stop()

        spiro_sound = random.choice(sound_files[pred_class])
        current_soundclass = spiro_sound.play() #play current sound class
    # pygame.mixer.music.play() #plays the random sound correlating to spirograph class
    #changed ^ because of slightly corrupted files

        
    #visualing video on screen with sound:
    # cv2.imshow('Spirograph', frame) - too dull of a green so manually made it green:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converts video to greyscale 
    mod_frame = cv2.merge([gray*0, gray, gray*0]) #uses green channel only 

    #adding background subtraction and canny edge detection as aesthetic filters
    fgMask = backSub.apply(frame) #applied background subtraction
    fgMask = cv2.medianBlur(fgMask, 9) #to reduce noise for edge detec. from backsub needed to add blur
    # fgMask = cv2.GaussianBlur(fgMask, (15,15), 0) #tested difference - does actually change it a lot when exaggerated settings
    edges = cv2.Canny(fgMask, 50, 150) #added canny edge detection to only effect the edges 
    edges_green = cv2.merge([edges * 0, edges, edges * 0]) #makes the edges green to match spirographs - BGR so keep green 

    mod_frame = cv2.addWeighted(mod_frame, 1.0, edges_green, 0.6, 0) #layers edges onto frame - so can see effects on screen 

    #adding detected class text description
    cv2.putText(mod_frame, f'Class: {pred_class}', (10, 30), #would remove text of pred class for visual aesthetic
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Spirograph', mod_frame) #had to make it green again bc opencv made it duller

    if cv2.waitKey(100) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()
