#!/usr/bin/env python
# coding: utf-8

# # DGMD S-17 Summer 2020 - Project PopBot
# 
# ## Team Members:
# >- Chris Crane
# >- Robert Clapp
# >- Andrew Pham
# >- James Sun
# 
# ## Project Objectives:
# **The objective of Team PupBot for this fianl DGMD S-17 Summer 2020 project include: 
# - Object Detection - Development of code to provide object detection on team member robots
# - Object Avoidance - Development of code to avoid specific objects 
# - Object Following - Development of code to follow specific objects
# - Object Retreat - Development of code to retreat or run from specific objects
# 
# ### Notes:
# ####**This notebook is not designed to be run in its entirety and kernel/robot restarts are required.** 
# 
# ####**Restart kernel between the running of each section - Kernel restarts will be annotated in section headings**
# 
# 1. Team PupBot's robots must first be trained to recognize objects in each team member's environment by running the code in the **Data Collection** section. 
# 2. All newly captured images in Data Collection must be saved to the `dataset` folder and subfolders.
# 3. After finishing data collection, run the **Train Model** section. Trained model data is saved in the `best_model.pth` file
# 4. After training the model, run the **Project Code Execute** section to demonstrate the objectives of the Team PupBot project.
# 5. It may be necessary to reboot the robot between data collection and the main entry point (if kernel restart does not work) to get the camera resource release

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
# ## Train Model (Kernel Reboot Required Here)
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


# ############################################################################################################
# ## Project Code Execute (Kernel Reboot and Possible Robot Reboot Required Here)
# ############################################################################################################

# In[1]:


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

# Load COCO object detection labels
with open('coco_label_map.json', 'r') as f:
    coco_labels = json.load(f)
f.close()

#for label in coco_labels:
#    print(str(label['id']) + ': ' + label['label'])


# In[2]:


model = ObjectDetector('ssd_mobilenet_v2_coco.engine')

camera = Camera.instance(width=300, height=300)

robot = Robot()
robot.stop;

heartbeat = Heartbeat()


# In[3]:


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


# In[5]:


blocked_widget = widgets.FloatSlider(min=0.0, max=1.0, value=0.0, description='blocked')
image_widget = widgets.Image(format='jpeg', width=300, height=300)
label_widget = widgets.IntText(value=1, description='tracked label') # Looking for a Person (value = 1)
speed_widget = widgets.FloatSlider(value=0.30, min=0.0, max=1.0, step=0.05, description='speed')
turn_gain_widget = widgets.FloatSlider(value=0.8, min=0.0, max=2.0, description='turn gain')
stop_button = widgets.Button(description='Stop', button_style='danger')
start_button = widgets.Button(description='Restart', button_style='success')
det_center_text = widgets.Text(value='')

display(widgets.VBox([
    widgets.HBox([image_widget, blocked_widget, stop_button, start_button]),
    widgets.HBox([label_widget,det_center_text]),
    speed_widget,
    turn_gain_widget
]))

width = int(image_widget.width)
height = int(image_widget.height)

def get_coco_label(label_id):
    label = 'UNKNOWN'
    for l in coco_labels:
        if l['id'] == label_id:
            label = l['label']
            break
    return label

#Computes the center x, y coordinates of the object
def detection_center(detection):
    bbox = detection['bbox']
    center_x = (bbox[0] + bbox[2]) / 2.0 - 0.5
    center_y = (bbox[1] + bbox[3]) / 2.0 - 0.5
    return (center_x, center_y)
    
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
        robot.left(0.4)
        image_widget.value = bgr8_to_jpeg(image)
        return
        
    # compute all detected objects
    detections = model(image)
    
    # draw all detections on image
    for det in detections[0]:
        bbox = det['bbox']
        cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])), (int(width * bbox[2]), int(height * bbox[3])), (255, 0, 0), 2)
        #cv2.putText(image, get_coco_label(det['label']), (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    
    # select detections that match selected class label
    matching_detections = [d for d in detections[0] if d['label'] == int(label_widget.value)]
    
    # get detection closest to center of field of view and draw it
    det = closest_detection(matching_detections)
    if det is not None:
        bbox = det['bbox']
        cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])), (int(width * bbox[2]), int(height * bbox[3])), (0, 255, 0), 5)
        #cv2.putText(image, get_coco_label(det['label']), (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    
    # otherwise go forward if no target detected
    if det is None:
        robot.forward(float(speed_widget.value))
        det_center_text.value = ''
        
    # otherwsie steer towards target
    else:
        # move robot forward and steer proportional target's x-distance from center
        center = detection_center(det)
        det_center_text.value = str(center)
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
        robot.stop()
        
stop_button.on_click(stop)
start_button.on_click(restart)

#execute({'new': camera.value})
camera.unobserve_all()
camera.observe(execute, names='value')
heartbeat.observe(handle_heartbeat_status, names='status')

