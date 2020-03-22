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

print ('Program started')

sim.simxFinish(-1)
clientID=sim.simxStart('127.0.0.1',19999,True,True,5000,5) 

if clientID!=-1:
    print ('Connected to remote API server')
    time.sleep(2)
else:
    sys.exit('Failed connecting to remote API server')

# Prepare initial values and retrieve handles:
wheelJoints = [-1,-1,-1,-1] # front left, rear left, rear right, front right
_, wheelJoints[0]=sim.simxGetObjectHandle(clientID,'rollingJoint_fl',sim.simx_opmode_blocking)
_, wheelJoints[1]=sim.simxGetObjectHandle(clientID,'rollingJoint_rl',sim.simx_opmode_blocking)
_, wheelJoints[2]=sim.simxGetObjectHandle(clientID,'rollingJoint_rr',sim.simx_opmode_blocking)
_, wheelJoints[3]=sim.simxGetObjectHandle(clientID,'rollingJoint_fr',sim.simx_opmode_blocking)
_, youBot = sim.simxGetObjectHandle(clientID,'youBot',sim.simx_opmode_blocking)
_, youBotRef = sim.simxGetObjectHandle(clientID,'youBot_ref',sim.simx_opmode_blocking)
_, tip = sim.simxGetObjectHandle(clientID,'youBot_positionTip',sim.simx_opmode_blocking)
_, target = sim.simxGetObjectHandle(clientID,'youBot_positionTarget',sim.simx_opmode_blocking)

# Start Driving
#sim.simxSetJointTargetVelocity(clientID, wheelJoints[0], 0.5, sim.simx_opmode_streaming)
#sim.simxSetJointTargetVelocity(clientID, wheelJoints[1], 0.5, sim.simx_opmode_streaming)
#sim.simxSetJointTargetVelocity(clientID, wheelJoints[2], 0.5, sim.simx_opmode_streaming)
#sim.simxSetJointTargetVelocity(clientID, wheelJoints[3], 0.5, sim.simx_opmode_streaming)
sim.simxSetJointTargetVelocity(clientID, tip, 1, sim.simx_opmode_streaming)
print('Start Driving')
# Now retrieve streaming data (i.e. in a non-blocking fashion):
startTime=time.time()
sim.simxGetIntegerParameter(clientID,sim.sim_intparam_mouse_x,sim.simx_opmode_streaming) # Initialize streaming
"""
while time.time()-startTime < 5:
    returnCode,data=sim.simxGetIntegerParameter(clientID,sim.sim_intparam_mouse_x,sim.simx_opmode_buffer) # Try to retrieve the streamed data
    if returnCode==sim.simx_return_ok: # After initialization of streaming, it will take a few ms before the first value arrives, so check the return code
        print ('Mouse position x: ',data) # Mouse position x is actualized when the cursor is over CoppeliaSim's window
        time.sleep(0.005)

    # Now send some data to CoppeliaSim in a non-blocking fashion:
    sim.simxAddStatusbarMessage(clientID,'Hello CoppeliaSim!',sim.simx_opmode_oneshot)
"""
# Before closing the connection to CoppeliaSim, make sure that the last command sent out had time to arrive. You can guarantee this with (for example):
sim.simxGetPingTime(clientID)
# Now close the connection to CoppeliaSim:
sim.simxFinish(clientID) 



print ('Program ended')

