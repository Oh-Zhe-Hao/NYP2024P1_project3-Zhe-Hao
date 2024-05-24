import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from controller import Robot, Camera, Keyboard

# Define YOLOv3 model implementation (assuming it's defined here)
class YOLOv3(torch.nn.Module):
    def __init__(self, num_classes):
        super(YOLOv3, self).__init__()
        # Define your YOLOv3 architecture here
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # Add more layers as needed

        # Calculate the expected output size from convolutional layers
        # Adjust this based on the architecture of your YOLOv3 model
        self.conv_output_size = self._calculate_conv_output_size()

        self.num_classes = num_classes
        self.classifier = torch.nn.Linear(self.conv_output_size, self.num_classes)

    def _calculate_conv_output_size(self):
        # Create a dummy input tensor to calculate the output size from conv layers
        x = torch.randn(1, 3, 416, 416)  # Assuming input size of (3, 416, 416)
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        # Add more convolutional layers and operations as needed
        conv_output_size = np.prod(x.size()[1:])  # Calculate the flattened size excluding batch dimension
        return conv_output_size

    def forward(self, x):
        # Implement YOLOv3 forward pass
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        # Add more forward pass logic for your YOLOv3 model

        x = x.view(x.size(0), -1)  # Flatten to (batch_size, conv_output_size)
        x = self.classifier(x)
        return x

# Instantiate the YOLOv3 model
model = YOLOv3(num_classes=80)  # Example: 80 classes for COCO dataset

# Load pretrained weights
model.load_state_dict(torch.load(r'C:\Users\ohzhe\OneDrive\Documents\Project\yolov3_dataset_last.pt', map_location=torch.device('cpu')), strict=False)
model.eval()  # Set model to evaluation mode

MAX_SPEED = 5.0  # Define max speed
SPEED_FACTOR = 1.0  # Speed factor for motor control

def convert_speed(speed):
    if speed < 0.2:
        return 0.0
    if speed > MAX_SPEED:
        speed = MAX_SPEED
    return SPEED_FACTOR * speed

def left(speed):
    s = convert_speed(speed)
    left_motor.setVelocity(-s)
    right_motor.setVelocity(s)

def right(speed):
    s = convert_speed(speed)
    left_motor.setVelocity(s)
    right_motor.setVelocity(-s)

def forward(speed):
    s = convert_speed(speed)
    left_motor.setVelocity(s)
    right_motor.setVelocity(s)

def backward(speed):
    s = convert_speed(speed)
    left_motor.setVelocity(-s)
    right_motor.setVelocity(-s)

def stop():
    left_motor.setVelocity(0.0)
    right_motor.setVelocity(0.0)

def process_camera_image(camera):
    try:
        if camera and camera.getWidth() > 0 and camera.getHeight() > 0:
            image = camera.getImage()

            # Convert the image to 3-channel (RGB) format
            image_np = np.frombuffer(image, np.uint8).reshape((camera.getWidth(), camera.getHeight(), 4))
            image_rgb = image_np[:, :, :3]  # Extract RGB channels (ignore alpha channel)

            image_pil = Image.fromarray(image_rgb)

            # Preprocess image for YOLOv3
            transform = transforms.Compose([
                transforms.Resize((416, 416)),
                transforms.ToTensor(),
            ])
            input_image = transform(image_pil).unsqueeze(0)

            return input_image
        else:
            print("Error: Camera not properly configured.")
            return None
    except Exception as e:
        print(f"Error in process_camera_image: {e}")
        return None

def handle_autonomous_control(image):
    with torch.no_grad():
        # Perform object detection using YOLOv3
        results = model(image)

        # Process detection results (example logic)
        # Assuming 'results' is a placeholder for inference results
        labels = ['1'] if np.random.rand() > 0.5 else []  # Example: Random detection
        # labels = results.names  # Use this line for real inference
        print(labels)

        # Example: Autonomous control based on detection
        if '1' in labels:
            backward(2)
            left(1)
        else:
            forward(3)

# Initialize robot, motors, camera, and keyboard
robot = Robot()
left_motor = robot.getDevice("left_wheel_hinge")
right_motor = robot.getDevice("right_wheel_hinge")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

camera = Camera("camera")
camera.enable(10)  # Enable the camera with a timestep of 10 ms

keyboard = Keyboard()
keyboard.enable(10)  # Enable the keyboard with a timestep of 10 ms

manual_control = True  # Flag for manual control

# Main loop
while robot.step(32) != -1:  # Use a timestep that matches camera frequency
    # Check for keyboard input
    key = keyboard.getKey()
    if key == ord('W'):
        forward(2)
        manual_control = True
    elif key == ord('S'):
        backward(2)
        manual_control = True
    elif key == ord('A'):
        left(1)
        manual_control = True
    elif key == ord('D'):
        right(1)
        manual_control = True
    elif key == ord(' '):  # Space bar to stop
        stop()
        manual_control = True
    else:
        manual_control = False

    # Process camera image
    input_image = process_camera_image(camera)

    # Handle autonomous control based on YOLOv3 detection
    if not manual_control and input_image is not None:
        handle_autonomous_control(input_image)

