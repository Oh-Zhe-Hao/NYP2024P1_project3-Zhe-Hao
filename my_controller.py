from controller import Robot, Keyboard, Camera, Display
import os

SPEED_FACTOR = 10.0
MAX_SPEED = 0.65

def convert_speed(speed):
    if speed < 0.2:
        return 0.0
    if speed > MAX_SPEED:
        speed = MAX_SPEED

    return SPEED_FACTOR * speed

def left(speed):
    s = convert_speed(speed)
    left_motor.setVelocity(s)
    right_motor.setVelocity(-s)

def right(speed):
    s = convert_speed(speed)
    left_motor.setVelocity(-s)
    right_motor.setVelocity(s)

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

def handle_keyboard_input(key):
    global camera_enabled
    if key == ord('W'):
        forward(0.5)
    elif key == ord('S'):
        backward(0.5)
    elif key == ord('D'):
        right(0.5)
    elif key == ord('A'):
        left(0.5)
    elif key == ord('P'):
        # Capture image from the camera
        image = camera.getImage()
        # Save the captured image
        filename = f"camera_image_{len(picture_count)}.png"
        camera.saveImage(filename, 100)
        picture_count.append(1) # Increment the picture count
    else:
        stop()

robot = Robot()

left_motor = robot.getDevice("left_wheel_hinge")
right_motor = robot.getDevice("right_wheel_hinge")
left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

keyboard = Keyboard()
timestep = int(robot.getBasicTimeStep())
keyboard.enable(timestep)  # Enable keyboard polling with the correct timestep

# Raspberry Pi Camera setup
camera = Camera("camera")
camera_enabled = False

# Display setup
display = Display("display")
display.setColor(0xFFFFFF)
display.setAlpha(0.7)

# Initialize picture count
picture_count = []

# Main loop
while robot.step(timestep) != -1:
    key = keyboard.getKey()
    handle_keyboard_input(key)

