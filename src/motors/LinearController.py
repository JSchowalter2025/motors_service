# -*- coding: utf-8 -*-
"""
Created on Fri Aug 07 17:13:32 2015

@author: qittlab
"""

from .APTMotor import APTMotor
class LinearController(APTMotor):
    
    def __init__(self, info):
        APTMotor.__init__(self,info['serial'], HWTYPE = 31)
        APTMotor.setVelocityParameters(self, info['minVel'], info['acc'], info['maxVel'])
        APTMotor.getStageAxisInformation(self)
        APTMotor.setStageAxisInformation(self, info['minPos'],info['maxPos'])
        self.attributes = info
        
    def mAbs(self, absPosition):
        pos = absPosition + self.attributes['zero']
        if pos > self.attributes['maxPos'] or pos < self.attributes['minPos']:
            print("-------------------------------")
            print("ERROR: Outside range of motor")
            print("-------------------------------")
            
            return "ERROR: Outside range of motor"
        else:
            APTMotor.mAbs(self,pos)
            return 'Success'
            
    def mRel(self, step):
        pos = self.getPos() + step + self.attributes['zero']
        if pos > self.attributes['maxPos'] or pos < self.attributes['minPos']:
            print("-------------------------------")
            print("ERROR: Outside range of motor")
            print("-------------------------------")
            return "ERROR: Outside range of motor"
            
        else:
            #might be a problem with controller's mRel linear range check and zero offset
            APTMotor.mRel(self,step)
            return "Success"
            
    def getPos(self):
        absolutePos = APTMotor.getPos(self)
        return absolutePos - self.attributes['zero']
    
        
if __name__ == '__main__':
    motor1 ={
    'name': 'motor1',
    'serial': 83842776,
    'minVel': 0.5,
    'maxVel': 1.0,
    'acc' : 1.0,
    'zero': 0}
    motor = LinearController(motor1)
    print('created')
