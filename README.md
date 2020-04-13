# Project Goal:
# Our P3DX ROBOT, combined with LBR R820 Robot Arm will navigate through the environment to find objects, which will be returned to a common area or detect walls, which then will move in a different direction.
# 
# Sense Walls:
# It will read values from the ultrasonic sensors to find the distance to the detected object. If the distance is less than 0.2, it has sensed a wall. 

# Current Implementation: 
# Robot arm uses inverse kinematics to move the joints and create a path using the handle of the box to be moved and the handle dummy of the return area. 

# Future Implementation:
# The robot arm will be mounted on the P3DX so that the P3DX can identify and stop where the object is detected. If the detected object handle matches what needs to be return, the arm will grab the object and return it to the specified area.   
