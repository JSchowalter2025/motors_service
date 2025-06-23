# -*- coding: utf-8 -*-
"""
Created on Thu May 29 15:20:39 2025

@author: qeiminipc
Jonathan Schowalter

Wrapper class to mimic the action of RotationController, but using the
roesel Elliptec library instead of APTmotor.
"""
from .elliptec.rotator import ELLRotator
from .elliptec.controller import ELLController #making my editor be quiet about the undefined objects
import elliptec

class RotationControllerELL(elliptec.Rotator):

    def __init__(self, info):
        ''' The APTmotors class is defined with an info dictionary. The elliptec motor class attempts
        to ask the motor itself for this information instead. I have commented out information which
        was previously defined by the constructor which is now read from the device: see ELLmotor.py
        '''    
        port = elliptec.find_ports()[0] #We need to find the port the motor is on to instance the controller.
                                        #TODO: this assumes the first port with an ELL on it is the one we want to control: bad!
        
        elliptec.Rotator.__init__(self, elliptec.Controller(port)) #info['serial'], HWTYPE=31)
        # APTMotor.setVelocityParameters(
        #   self, info['minVel'], info['acc'], info['maxVel'])
        #self.set_backlash_dist(1.00019)  # sets the backlash correction
        '''
        The backlash correction method in APTmotors was custom from our lab and doesn't exist in
        the elliptec package. I think ELL motors may do this on their own, but we may have to
        reimplement the method it if it proves necessary.'''
        
        self.attributes = info #We still need this because it contains self.attributes['zero']
        
        #self.linear_range = (0, 360)

    def mAbs(self, absPosition):
        pos = (absPosition + self.attributes['zero']) % 360
        # print("Moving to %r (%r)..."%(absPosition,pos))
        try:
            elliptec.Rotator.set_angle(self, pos)
        except Exception:
            print('Failed')
            return 'Failed'
        # print('\t Moved to %r'%self.getPos())
        return 'Success'

    def mRel(self, step):
        self.shift_angle(step)
        return 'Success'

    def mHome(self):
        elliptec.Rotator.home(self) #, velocity=9.99978, offset=4.00023) ?
        return 'Success'

    def getPos(self):
        absolutePos = elliptec.Rotator.get_angle(self)
        return (360 + absolutePos - self.attributes['zero']) % 360

    def getAPos(self):
        return elliptec.Rotator.get_angle(self)


#The following information is irrelevant to this class , it is for the APT motors.
''' 
if __name__ == '__main__':
    motor1 = {
        'name': 'motor1',
        'serial': 83842776,
        'minVel': 0.5,
        'maxVel': 1.0,
        'acc': 1.0,
        'zero': 0}
    motor2 = {
        'name': 'motor2',
        'serial': 83854943,
        'minVel': 0.5,
        'maxVel': 1.0,
        'acc': 1.0,
        'zero': 160.59}
    BOBHWP2 = {
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
    motor = RotationControllerELL(BOBHWP1)
    print('Motor object created: ', motor)'''
