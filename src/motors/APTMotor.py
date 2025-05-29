from __future__ import absolute_import
from __future__ import print_function
from .pyAPT import Message, Controller, MTS50, PRM1, OutOfRangeError
#from pyAPT import Message, Controller, MTS50, PRM1, OutOfRangeError

import time
import pylibftdi
import sys

"""
A wrapper class to provide interface between bell_motors
and linux-compatible pyAPT's "Controller"

Linux-compatible pyAPT found here: https://github.com/freespace/pyAPT
Modifications were made as necessary

Collin Schlager
"""



class APTMotor(Controller):
    def __init__(self, SerialNumber=None, label=None, HWTYPE=31):
        super(APTMotor, self).__init__(serial_number=SerialNumber, label=label)
        
        #NOTE: This is specific for a PRM1.
        
        # http://www.thorlabs.de/newgrouppage9.cfm?objectgroup_id=2875
        # Note that these values should be pulled from the APT User software,
        # as they agree with the real limits of the stage better than
        # what the website or the user manual states
        self.max_velocity = 0.3 # units?
        self.max_acceleration = 0.3 # units?

        # from the manual
        # encoder counts per revoultion of the output shaft: 34304
        # no load speed: 16500 rpm = 275 1/s
        # max rotation velocity: 25deg/s
        # Gear ratio: 274 / 25 rounds/deg
        # to move 1 deg: 274/25 rounds = 274/25 * 34304 encoder steps
        # measured value: 1919.2689
        # There is an offset off 88.2deg -> enc(0) = 88.2deg
        enccnt = 1919.2698

        T = 2048/6e6

        # these equations are taken from the APT protocol manual
        self.position_scale = enccnt  #the number of enccounts per deg
        self.velocity_scale = enccnt * T * 65536
        self.acceleration_scale = enccnt * T * T * 65536

        self.linear_range = (-180, 180)
            
    def mAbs(self, absPosition):
        self.goto(float(absPosition))

    def moveOptimal(self, destination):
        """Rotates to destination position via the optimal path: CCW or CW
        See controller --> move() edits for more details"""
        curposition = self.position()
        absDelta = destination - curposition
        relDelta = absDelta%360
        if relDelta >= 180:
            optimal = (360-relDelta)*-1
        else:
            optimal = relDelta
        self.move(optimal)
        
    def mRel(self, step):
        self.move(step)
        
    def getPos(self):
        return self.position()
        
    def getStageAxisInformation(self):
        return [self.linear_range[0], self.linear_range[1]]
        
    def setStageAxisInformation(self, minimumPosition, maximumPosition):
        self.linear_range = (minimumPositition, maximumPosition)
        
        
    def setVelocityParameters(self, minVel, acc, maxVel):
        """Not quite robust right now
        Needs to set minVel"""
        self.max_veloicty = maxVel
        self.max_acceleration = acc
        return True
        
    def getStatus(self):
        status = self.status()
        print('\tController status:')
        print('\t\tPosition: %.3fmm (%d cnt)'%(status.position, status.position_apt))
        print('\t\tVelocity: %.3fmm'%(status.velocity))
        print('\t\tStatus:',status.flag_strings())
        return status
        

if __name__ == "__main__":
    me = APTMotor(SerialNumber=83854955)
