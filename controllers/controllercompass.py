from controller import Robot, Compass, Motor, Supervisor, Accelerometer

import matplotlib.pyplot as plt



robot = Supervisor()

timestep = int(robot.getBasicTimeStep())



# Compass

compass = Compass("compass")

if compass is None:

    print("Could not find a compass.")

    exit()

compass.enable(100)





# MOTORS

# Find motors and set them up to velocity control

left_motor = Motor("wheel_left_joint")

right_motor = Motor("wheel_right_joint")



if left_motor is None or right_motor is None:

    print("Could not find motors.")

    exit()

left_motor.setPosition(float("inf"))

right_motor.setPosition(float("inf"))




 
while robot.step(timestep) != -1:

    left_motor.setVelocity(0.0)

    right_motor.setVelocity(6.0)
     
    # Read raw compass data
    compass_data = compass.getValues()
    
    x = compass_data[0]  
    y = compass_data[1]
    
    # Calculate the heading angle (in degrees) from the compass data
        heading_radians = math.atan2(compass_data[0], compass_data[2])
heading_degrees = math.degrees(heading_radians)
    
    # Print compass data
    print(f"Compass Data:[{x} , {y}]")