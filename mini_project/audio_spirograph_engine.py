import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from PIL import Image, ImageDraw
import math
import os

FS = 44100
DURATION = 2  #seconds per sample
REPEATS_OVAL = 50 #frames
STEPS_OVAL = 360 #circular for spirograph pattern
X_RADIUS_BASE = 40 #modifiable x
Y_RADIUS_BASE = 180 #modifiable y
ROTATION_PER_SHAPE = 5 #how far apart each oval is - could be modifiable from sound
IMG_SIZE = 600 

BASE_PATH = "mini_project"

#make classes for each drawing to go into 
classes = ["high_loud","high_quiet","low_loud","low_quiet"]
for c in classes:
    os.makedirs(os.path.join(BASE_PATH, "dataset2/train", c), exist_ok=True) #adds to 'train'
    os.makedirs(os.path.join(BASE_PATH, "dataset2/sounds2", c), exist_ok=True) #adds matching sound to same classes in sounds2

#need to count samples to save in order with matching sound
sample_counts = {}
for c in classes:
    folder = os.path.join(BASE_PATH, "dataset2/train", c)
    existing = [f for f in os.listdir(folder) if f.endswith(".png")]
    sample_counts[c] = len(existing)


#deconstructing audio 
### help from chatgpt & stack overflow to extract audio using numpy :
def get_pitch_autocorr(x):
    x = x - np.mean(x)
    corr = np.correlate(x, x, mode='full')
    corr = corr[len(corr)//2:]
    d = np.diff(corr)
    start = np.where(d > 0)[0][0]
    peak = np.argmax(corr[start:]) + start
    return 0 if peak==0 else FS/peak

def map_value(val, in_min, in_max, out_min, out_max):
    val = max(min(val, in_max), in_min)
    return out_min + (out_max-out_min)*(val-in_min)/(in_max-in_min)
###


#defining boundaries for classes
def get_class_label(mean_pitch, mean_amp):
    pitchLine = "high" if mean_pitch>600 else "low"  #defining boundary line for pitch - was going to do female vs male voice but 300 too low
    ampLine = "loud" if mean_amp>0.02 else "quiet" #and for amplitude 
    return f"{pitchLine}_{ampLine}"

#spirograph drawing on PIL
def draw_spirograph_pil(pitches, amps, size=IMG_SIZE):
    spirograph = Image.new("RGB", (size, size), "black")
    draw = ImageDraw.Draw(spirograph) 
    center = size // 2

    #for every oval...
    for r in range(REPEATS_OVAL):
        #low pitch = rounder, high pitch = spikey 
        x_radius = X_RADIUS_BASE * map_value(pitches[r], 100, 1000, 3.0, 0.1) #mapping to more extreme parameters
        #quiet = smaller, louder = bigger/taller
        y_radius = Y_RADIUS_BASE * map_value(amps[r], 0.001, 0.05, 0.5, 2.0) 

        angle_offset = math.radians(r * ROTATION_PER_SHAPE) 
        points = []

        for i in range(STEPS_OVAL):
            angle = math.radians(i)
            x = x_radius * math.cos(angle)
            y = y_radius * math.sin(angle)
            # rotate oval
            x_rot = x*math.cos(angle_offset) - y*math.sin(angle_offset)
            y_rot = x*math.sin(angle_offset) + y*math.cos(angle_offset)
            points.append((center + x_rot, center + y_rot))

        draw.line(points, fill=(34,139,34), width=1)

    return spirograph


while True:
    input("press ENTER to record 2s audio")
    print("recording...")
    audio = sd.rec(int(DURATION*FS), samplerate=FS, channels=1)
    sd.wait()
    audio = audio[:,0]

    #split audio into frames
    frame_size = len(audio) // REPEATS_OVAL
    audio_frames = [audio[i*frame_size:(i+1)*frame_size] for i in range(REPEATS_OVAL)]

    #compute per-frame amplitude and pitch
    amps = [np.sqrt(np.mean(f**2)) for f in audio_frames]
    pitches = [get_pitch_autocorr(f) for f in audio_frames]

    mean_amp = np.mean(amps)
    mean_pitch = np.mean(pitches)
    label = get_class_label(mean_pitch, mean_amp)
    idx = sample_counts[label]
    print(f"mean pitch={mean_pitch:.1f} Hz | mean amp={mean_amp:.4f} | class={label} | index={idx}")

    #draw spirograph directly to PIL image
    img = draw_spirograph_pil(pitches, amps)
    img_path = os.path.join(BASE_PATH, "dataset2/train", label, f"{label}_{idx}.png")
    img.save(img_path)
    print("saved image:", img_path)

    #save audio with same name
    audio_path = os.path.join(BASE_PATH, "dataset2/sounds2", label, f"{label}_{idx}.wav")
    write(audio_path, FS, audio.astype(np.float32))
    print("saved audio:", audio_path)

    sample_counts[label] += 1 
