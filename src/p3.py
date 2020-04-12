# Make sure to have the server side running in CoppeliaSim:
# in a child script of a CoppeliaSim scene, add following command
# to be executed just once, at simulation start:
#
# simRemoteApi.start(19999)
#
# then start simulation, and run this program.
#
# IMPORTANT: for each successful call to simxStart, there
# should be a corresponding call to simxFinish at the end!

try:
    import sim
except:
    print("Cannot import sim")

import time
import sys
import numpy as np

print ('Program started')

sim.simxFinish(-1)
clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5)

if clientID!=-1:
    print ('Connected to remote API server')
    time.sleep(2)
else:
    sys.exit('Failed connecting to remote API server')

# Prepare initial values and retrieve handles:
usensors = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
_, leftMotor = sim.simxGetObjectHandle(clientID,'Pioneer_p3dx_leftMotor',sim.simx_opmode_blocking)
_, rightMotor = sim.simxGetObjectHandle(clientID,'Pioneer_p3dx_rightMotor',sim.simx_opmode_blocking)

usensor = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
for i in range(len(usensor)):
    _, usensor[i] = sim.simxGetObjectHandle(clientID, 'Pioneer_p3dx_ultrasonicSensor'+str(i+1), sim.simx_opmode_streaming)

"""
function sysCall_init()
    usensors={-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
    for i=1,16,1 do
        usensors[i]=sim.getObjectHandle("Pioneer_p3dx_ultrasonicSensor"..i)
    end

    noDetectionDist=0.5
    maxDetectionDist=0.2
    detect={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
    braitenbergL={-0.2,-0.4,-0.6,-0.8,-1,-1.2,-1.4,-1.6, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}
    braitenbergR={-1.6,-1.4,-1.2,-1,-0.8,-0.6,-0.4,-0.2, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}
    v0=2
end
"""

# Start Driving
sim.simxSetJointTargetVelocity(clientID, leftMotor, 0.7, sim.simx_opmode_streaming)
sim.simxSetJointTargetVelocity(clientID, rightMotor, 0.7, sim.simx_opmode_streaming)
print('Start Driving')

detectionState = [False]*16

for i in range(16):
    _, usensor[i] = sim.simxGetObjectHandle(clientID, 'Pioneer_p3dx_ultrasonicSensor'+str(i+1), sim.simx_opmode_blocking)
    _, detectionState[i], detectionPoint, detectedObjectHandle, detectedSurfaceNormalVector =  sim.simxReadProximitySensor(clientID, usensor[i], sim.simx_opmode_streaming)
_, targetPosition = sim.simxGetJointPosition(clientID, leftMotor,sim.simx_opmode_streaming)

minDetection = 0.4
maxDetection = 0.5
stepSize = 1.0

while(True):
    
    senseDistance = np.array([])
    
    for i in range(16):
        
        _, usensor[i] = sim.simxGetObjectHandle(clientID, 'Pioneer_p3dx_ultrasonicSensor'+str(i+1), sim.simx_opmode_blocking)
        _, detectionState[i], detectionPoint, detectedObjectHandle, detectedSurfaceNormalVector =  sim.simxReadProximitySensor(clientID, usensor[i], sim.simx_opmode_buffer)

        if detectionState[i]==False:
            distance = maxDetection
        else:
            distance = np.linalg.norm(detectionPoint)
        senseDistance = np.append(senseDistance, distance)

    if np.min(senseDistance) < minDetection:
        sim.simxSetJointTargetVelocity(clientID, leftMotor, 0, sim.simx_opmode_streaming)
        sim.simxSetJointTargetVelocity(clientID, rightMotor, 0, sim.simx_opmode_streaming)
        break

# Now retrieve streaming data (i.e. in a non-blocking fashion):
startTime=time.time()

# Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
sim.simxGetPingTime(clientID)

# Now close the connection to CoppeliaSim:
sim.simxFinish(clientID)


print ('Program ended')


"""
    function sysCall_init()
    usensors={-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1}
    for i=1,16,1 do
    usensors[i]=sim.getObjectHandle("Pioneer_p3dx_ultrasonicSensor"..i)
    end
    motorLeft=sim.getObjectHandle("Pioneer_p3dx_leftMotor")
    motorRight=sim.getObjectHandle("Pioneer_p3dx_rightMotor")
    noDetectionDist=0.5
    maxDetectionDist=0.2
    detect={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
    braitenbergL={-0.2,-0.4,-0.6,-0.8,-1,-1.2,-1.4,-1.6, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}
    braitenbergR={-1.6,-1.4,-1.2,-1,-0.8,-0.6,-0.4,-0.2, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}
    v0=2
    end
    -- This is a very simple EXAMPLE navigation program, which avoids obstacles using the Braitenberg algorithm
    
    
    function sysCall_cleanup()
    
    end
    
    function sysCall_actuation()
    for i=1,16,1 do
    res,dist=sim.readProximitySensor(usensors[i])
    if (res>0) and (dist<noDetectionDist) then
    if (dist<maxDetectionDist) then
    dist=maxDetectionDist
    end
    detect[i]=1-((dist-maxDetectionDist)/(noDetectionDist-maxDetectionDist))
    else
    detect[i]=0
    end
    end
    
    vLeft=v0
    vRight=v0
    
    for i=1,16,1 do
    vLeft=vLeft+braitenbergL[i]*detect[i]
    vRight=vRight+braitenbergR[i]*detect[i]
    end
    
    sim.setJointTargetVelocity(motorLeft,vLeft)
    sim.setJointTargetVelocity(motorRight,vRight)
    end

"""
