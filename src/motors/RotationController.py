# -*- coding: utf-8 -*-
"""
Created on Wed Aug 05 12:10:18 2015

@author: qittlab
Collin Schlager
"""
from APTMotor import APTMotor
#from .APTMotor import APTMotor

class RotationController(APTMotor):

    def __init__(self, info):
        APTMotor.__init__(self,info['serial'], HWTYPE = 31)
        APTMotor.setVelocityParameters(self,info['minVel'],info['acc'],info['maxVel'])
        self.set_backlash_dist(1.00019) #sets the backlash correction
        self.attributes = info
        self.linear_range = (0, 360)

    def mAbs(self, absPosition):
        pos = (absPosition + self.attributes['zero'])%360
        # print("Moving to %r (%r)..."%(absPosition,pos))
        try:
            APTMotor.moveOptimal(self,pos)
        except:
            print('Failed')
            return 'Failed'
        # print('\t Moved to %r'%self.getPos())
        return 'Success'

    def mRel(self, step):
        self.move(step)
        return 'Success'

    def mHome(self):
        APTMotor.home(self, velocity=9.99978, offset=4.00023)
        return 'Success'

    def getPos(self):
        absolutePos = APTMotor.getPos(self)
        return (360 + absolutePos - self.attributes['zero']) % 360

    def getAPos(self):
        return APTMotor.getPos(self)


if __name__ == '__main__':
    motor1 ={
    'name': 'motor1',
    'serial': 83842776,
    'minVel': 0.5,
    'maxVel': 1.0,
    'acc' : 1.0,
    'zero': 0}
    motor2 ={
    'name': 'motor2',
    'serial': 83854943,
    'minVel': 0.5,
    'maxVel': 1.0,
    'acc' : 1.0,
    'zero': 160.59}
    BOBHWP2 ={
    'acc': 25,
    'maxVel': 25,
    'minVel': 5,
    'name': "BobHWP2",
    'serial': 83854943,
    'zero': 16.7,
    'type': "rotational"}
    BOBHWP1 = {
    'acc': 25,
    'maxVel': 25,
    'minVel': 5,
    'name': 'BobHWP1',
    'serial': 83857280,
    'zero': -3.39,
    'type': 'rotational'}
#
    motor = RotationController(BOBHWP1)
    print('Motor object created: ', motor)
