#!/usr/bin/env python
# coding: utf-8

# # DGMD S-17 Summer 2020 - Project PupBot
# 
# ## Team Members:
# >- Chris Crane
# >- Robert Clapp
# >- Andrew Pham
# >- James Sun
# 
# ## Project Objectives:
# **Enter Project Objectives Here
# 
# ### Notes:
# ####**Do not run the notebook in its entirety.** 
# 
# ####**Restart kernel between the running of each section**
# 
# 1. The robot will need to be trained for each member's environment using the **Data Collection** section. 
# 2. All new captured images should be saved to the `dataset` folder.
# 3. After finishing data collection, run the **Train Model** section. The trained model should be saved in the file `best_model.pth`
# 4. After finishing training the model, run the **Main Entry Point** section.
# 5. It may be necessary to reboot the robot between data collection and the main entry point to get the camera resource release

# #######################################################################################
# ## Main Entry Point
# #######################################################################################

# In[ ]:


# Library Imports
import torch
import torchvision
import torch.nn.functional as F
import cv2
import numpy as np
from IPython.display import display
import ipywidgets.widgets as widgets
import traitlets
from jetbot import Heartbeat, ObjectDetector, Camera, Robot, bgr8_to_jpeg
import json
import time

# Load COCO object detection labels
with open('coco_label_map.json', 'r') as f:
    coco_labels = json.load(f)
f.close()


# In[ ]:


model = ObjectDetector('ssd_mobilenet_v2_coco.engine')

camera = Camera.instance(width=300, height=300, fps=6)

robot = Robot()
robot.stop() # Sometimes when the Robot instance is created, the robot starting moving

heartbeat = Heartbeat()


# In[ ]:


collision_model = torchvision.models.alexnet(pretrained=False)
collision_model.classifier[6] = torch.nn.Linear(collision_model.classifier[6].in_features, 2)
collision_model.load_state_dict(torch.load('best_model.pth'))
device = torch.device('cuda')
collision_model = collision_model.to(device)

mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])

normalize = torchvision.transforms.Normalize(mean, stdev)

def preprocess(camera_value):
    global device, normalize
    x = camera_value
    x = cv2.resize(x, (224, 224))
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    return x


# In[ ]:


import random as rnd

blocked_widget = widgets.FloatSlider(min=0.0, max=1.0, value=0.0, description='blocked')
image_widget = widgets.Image(format='jpeg', width=300, height=300)
label_widget = widgets.IntText(value=1, description='tracked label') # Looking for a Person (value = 1)
speed_widget = widgets.FloatSlider(value=0.45, min=0.0, max=1.0, step=0.05, description='speed')
turn_gain_widget = widgets.FloatSlider(value=0.6, min=0.0, max=2.0, description='turn gain')
stop_button = widgets.Button(description='Stop', button_style='danger')
start_button = widgets.Button(description='Restart', button_style='success')
det_center_text = widgets.Text(value='')
det_confidence_text = widgets.Text(value='')
det_label_text = widgets.Text(value='')
dist_text = widgets.Text(value='')

display(widgets.VBox([
    widgets.HBox([image_widget, widgets.VBox([blocked_widget, widgets.HBox([stop_button, start_button])])]),
    widgets.HBox([label_widget, det_center_text, dist_text]),
    widgets.HBox([speed_widget, det_confidence_text]),
    widgets.HBox([turn_gain_widget, det_label_text])
]))

width = int(image_widget.width)
height = int(image_widget.height)

#Returns the COCO label given label id
def get_coco_label(label_id):
    label = 'UNKNOWN'
    for l in coco_labels:
        if l['id'] == label_id:
            label = l['label']
            break
    return label

#Turns robot 90 degrees
def turn_90_degrees():
    robot.left(float(speed_widget.value))
    time.sleep(0.50)

#Determine if the detection found is one being looked for and meets the confidence threshold
def is_matching_detection(detection):
    return detection['label'] == int(label_widget.value) and detection['confidence'] > 0.70 
    
#Computes the center x, y coordinates of the object
def detection_center(detection):
    bbox = detection['bbox']
    center_x = (bbox[0] + bbox[2]) / 2.0 - 0.5
    center_y = (bbox[1] + bbox[3]) / 2.0 - 0.5
    return (center_x, center_y)

#Calculates the distance to detection
def distance_to_detection(detection):
    (x1, y1, x2, y2) = detection['bbox']
    fl = 1.85 #focal length mm
    sh = 75 #sensor height mm
    ph = 1700 #average human height mm
    ih = height #image height pixels
    oh = (y2 - y1) * height #object height in pixels
    return (fl * ph * height) / (oh * sh)

#Determines if the robot is too close to the detected object
def is_too_close(detection):
    return distance_to_detection(detection) < 100

#Moves the robot away from the detected object
def run_away():
    det_label_text.value = det_label_text.value + ': Running Away!!'
    turn_90_degrees()
    robot.forward(float(speed_widget.value * 1.5))

#Computes the length of the 2D vector
def norm(vec):
    return np.sqrt(vec[0]**2 + vec[1]**2)

#Finds the detection closest to the image center
def closest_detection(detections):
    closest_detection = None
    for det in detections:
        center = detection_center(det)
        if closest_detection is None:
            closest_detection = det
        elif norm(detection_center(det)) < norm(detection_center(closest_detection)):
            closest_detection = det
    return closest_detection

#Process camera input, find detections, follow target if identidied
def execute(change):
    image = change['new']
    
    # execute collision model to determine if blocked
    collision_output = collision_model(preprocess(image)).detach().cpu()
    prob_blocked = float(F.softmax(collision_output.flatten(), dim=0)[0])
    blocked_widget.value = prob_blocked
    
    # turn left if blocked
    if prob_blocked > 0.5:
        robot.left(float(speed_widget.value))
        image_widget.value = bgr8_to_jpeg(image)
        return
        
    # compute all detected objects
    detections = model(image)
    
    # draw all detections on image
    for det in detections[0]:
        if det['confidence'] > 0.5:
            (x, y, w, h) = det['bbox']
            l = get_coco_label(det['label'])
            cv2.rectangle(image, (int(width * x), int(width * y)), (int(width * w), int(height * h)), (255, 0, 0), 2)
            cv2.putText(image, l, (int(width * x), int(height * y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
    
    # get detection closest to center of field of view that match selected class label and draw it
    det = closest_detection([d for d in detections[0] if is_matching_detection(d)])
    
    if det is None:
        # otherwise go forward if no target detected
        robot.forward(float(speed_widget.value))
    else:
        (x, y, w, h) = det['bbox']
        l = get_coco_label(det['label'])
        cv2.rectangle(image, (int(width * x), int(height * y)), (int(width * w), int(height * h)), (0, 255, 0), 2)
        cv2.putText(image, l, (int(width * x), int(height * y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # move robot forward and steer proportional target's x-distance from center
        center = detection_center(det)
        det_center_text.value = str(center)
        det_confidence_text.value = str(det['confidence'])
        det_label_text.value = l
        dist_text.value = str(distance_to_detection(det))
        if is_too_close(det) == True:
            run_away()
        else:
            robot.set_motors(
                float(speed_widget.value + turn_gain_widget.value * center[0]),
                float(speed_widget.value - turn_gain_widget.value * center[0])
            )
    
    # update image widget
    image_widget.value = bgr8_to_jpeg(image)
    
def stop(change):
    camera.unobserve_all()
    time.sleep(0.5)
    robot.stop()

def restart(change):
    camera.unobserve_all()
    camera.observe(execute, names='value')

# this function will be called when heartbeat 'alive' status changes
def handle_heartbeat_status(change):
    if change['new'] == Heartbeat.Status.dead:
        #stop(change)
        det_center_text.value = 'Lost Connection'
        
stop_button.on_click(stop)
start_button.on_click(restart)

execute({'new': camera.value})
camera.unobserve_all()
camera.observe(execute, names='value')
heartbeat.observe(handle_heartbeat_status, names='status')


# ##############################################################################
# ## Data Collection
# ##############################################################################

# In[ ]:


# Display live camera feed
import traitlets
import ipywidgets.widgets as widgets
from IPython.display import display
from jetbot import Camera, bgr8_to_jpeg
from uuid import uuid1
import os

camera = Camera.instance(width=224, height=224)

image = widgets.Image(format='jpeg', width=224, height=224)  # this width and height doesn't necessarily have to match the camera

camera_link = traitlets.dlink((camera, 'value'), (image, 'value'), transform=bgr8_to_jpeg)


# In[ ]:


blocked_dir = 'dataset/blocked'
free_dir = 'dataset/free'

# we have this "try/except" statement because these next functions can throw an error if the directories exist already
try:
    os.makedirs(free_dir)
    os.makedirs(blocked_dir)
except FileExistsError:
    print('Directories not created becasue they already exist')


# In[ ]:


button_layout = widgets.Layout(width='128px', height='64px')
free_button = widgets.Button(description='add free', button_style='success', layout=button_layout)
blocked_button = widgets.Button(description='add blocked', button_style='danger', layout=button_layout)
free_count = widgets.IntText(layout=button_layout, value=len(os.listdir(free_dir)))
blocked_count = widgets.IntText(layout=button_layout, value=len(os.listdir(blocked_dir)))


# In[ ]:


def save_snapshot(directory):
    image_path = os.path.join(directory, str(uuid1()) + '.jpg')
    with open(image_path, 'wb') as f:
        f.write(image.value)

def save_free():
    global free_dir, free_count
    save_snapshot(free_dir)
    free_count.value = len(os.listdir(free_dir))
    
def save_blocked():
    global blocked_dir, blocked_count
    save_snapshot(blocked_dir)
    blocked_count.value = len(os.listdir(blocked_dir))
    
free_button.on_click(lambda x: save_free())
blocked_button.on_click(lambda x: save_blocked())

display(image)
display(widgets.HBox([free_count, free_button]))
display(widgets.HBox([blocked_count, blocked_button]))


# #######################################################################################
# ## Train Model
# #######################################################################################

# In[ ]:


import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms


# In[ ]:


# Create dataset instance
dataset = datasets.ImageFolder(
    'dataset',
    transforms.Compose([
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
)


# In[ ]:


# Split dataset into train and test sets
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - 50, 50])


# In[ ]:


# Create data loaders to load data in batches
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4
)


# In[ ]:


# Define the neural network
model = models.alexnet(pretrained=True)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
device = torch.device('cuda')
model = model.to(device)


# In[ ]:


# Train the neural network
NUM_EPOCHS = 30
BEST_MODEL_PATH = 'best_model.pth'
best_accuracy = 0.0

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(NUM_EPOCHS):
    
    for images, labels in iter(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()
    
    test_error_count = 0.0
    for images, labels in iter(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        test_error_count += float(torch.sum(torch.abs(labels - outputs.argmax(1))))
    
    test_accuracy = 1.0 - float(test_error_count) / float(len(test_dataset))
    print('%d: %f' % (epoch, test_accuracy))
    if test_accuracy > best_accuracy:
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        best_accuracy = test_accuracy

