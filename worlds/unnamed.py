from controller import Robot, Compass, Motor, Accelerometer, Gyro

import math

import matplotlib.pyplot as plt

robot = Robot()

timestep = int(robot.getBasicTimeStep())

# Compass

compass = Compass("compass")

accelerometer = Accelerometer("accelerometer")

gyro = Gyro("gyro")

x_compass = []  # To store timestamps

y_compass = []  # To store filtered compass sensor values

if compass is None:
    print("Could not find a compass.")

    exit()

compass.enable(timestep)

accelerometer.enable(timestep)

gyro.enable(timestep)

# MOTORS

left_motor = Motor("wheel_left_joint")

right_motor = Motor("wheel_right_joint")

if left_motor is None or right_motor is None:
    print("Could not find motors.")

    exit()

left_motor.setPosition(float("inf"))

right_motor.setPosition(float("inf"))

# Filter parameters

filter_size = 10  # Adjust the filter size as needed

compass_data_buffer = []

# Create the figure for plotting outside the loop

plt.figure(figsize=(10, 6))

plt.xlabel('Time (seconds)')

plt.ylabel('Compass Data (Radians)')

plt.title('Filtered Compass Data Over Time')

plt.grid(True)

while robot.step(timestep) != -1:

    left_motor.setVelocity(0.0)

    right_motor.setVelocity(6.0)

    # Update sensor values inside the loop

    north = compass.getValues()

    gyro_data = gyro.getValues()

    acc_data = accelerometer.getValues()

    rad = math.atan2(north[0], north[1])

    bearing = (rad - 1.5708) / math.pi * 180.0

    bearing = abs(bearing)

    if bearing < 10.0 or bearing >= 350.0:

        direction = "N"

    elif 70.0 <= bearing < 110.0:

        direction = "E"

    elif 160.0 <= bearing < 200.0:

        direction = "S"

    elif 250.0 <= bearing < 290.0:

        direction = "W"

    else:

        direction = "Unknown"

        # Filter compass data with a simple moving average filter

    compass_data_buffer.append(rad)

    if len(compass_data_buffer) > filter_size:
        compass_data_buffer.pop(0)

    filtered_compass_rad = sum(compass_data_buffer) / len(compass_data_buffer)

    # Convert filtered compass data back to x and y components

    filtered_north_x = math.sin(filtered_compass_rad)

    filtered_north_y = math.cos(filtered_compass_rad)

    time_in_seconds = robot.getTime()

    # Store data in lists

    x_compass.append(time_in_seconds)

    y_compass.append(filtered_compass_rad)

    # Add motor control or other processing here if needed

    # Update the plot with the new data

    plt.clf()  # Clear the previous plot

    plt.plot(x_compass, y_compass, label='Filtered Compass Data (Radians)')

    plt.pause(0.01)  # Pause to allow the plot to update

# Print gyro and accelerometer data

print("Gyro Data:", gyro.getValues())

print("Accelerometer Data:", accelerometer.getValues())

# Show the final plot after the simulation completes

plt.show() 