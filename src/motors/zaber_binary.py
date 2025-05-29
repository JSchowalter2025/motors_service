from sys import platform

import serial
import struct
import time
from zaber.serial import BinarySerial, BinaryDevice, BinaryCommand, BinaryReply

"""
zaber motor library using zaber.serial:
http://www.zaber.com/support/docs/api/core-python/0.8.1/

Important notes:
- Are the replies for data returning the actual data?
- T-series only supports binary protocol (no ASCII)
"""

class zaber_base():
    def __init__(self, port, devices = [1],timeout = 1, chNames = ''):
        self.port_number = port
        self.struct = struct.Struct('<2Bi')
        self.packet_size = self.struct.size
        self.ser = self.connect(self.port_number, timeout)
        print(self.ser)
        self.devices = self.init_devices(devices) #creates BinaryDevice from each # in devices

        self.potentiometer = 0
        self.backlash = 1
        self.stiction = 1
        self.LED = 1
        self.set_speed_all(.8)
#        knob_speed = 0.05 #deg/s

        #for i in devices:
        #    self.backlash_correction(i, self.backlash)
#            self.set_speed_knob(i, knob_speed)
#            self.set_speed_profile_knob(i, 1) # linear response
    """
        count = 0
        while(len(self.ser.read(50))!=0):
            count += 1
            returnMsgCoded = self.ser.read(self.packet_size)
            returnMsg = self.struct.unpack(returnMsgCoded)
            self.ser.flushInput()
            self.ser.flushOutput()
            print(str(count), str(returnMsg))
    """

    def init_devices(self, devices):
        binaryDevices = {}
        for device in devices:
            # print(device)
            bDevice = BinaryDevice(self.ser, device)
            # print(bDevice)

            binaryDevices[device]=bDevice
        return binaryDevices

    def connect(self, port, timeout=1):
        if platform == "linux" or platform == "linux2":
            port = "/dev/ttyUSB"+str(port)
            port = str(port)
        elif platform == "win32":
            port = "COM"+str(port)
            port = str(port)
        else:
            print("Platform %r not recognized." % platform)
            return -1
        try:
            print("Connecting to port: ", port)
            ser = BinarySerial(port) #can we add a timeout?
        except Exception as error:
            print("Could not connect to port", port)
            print("Zaber motors failed: ", error)
            ser = -1
        return ser

    def close(self):
        try:
            self.ser.close()
        except:
            print("Could not close port", self.port_number)

    # Helper to check that commands succeeded.
    def check_command_succeeded(self,reply):
        """
        Return true if command succeeded, print reason and return false if command
        rejected
        """
        if reply.command_number == 255: # 255 is the binary error response code.
            # print ("Command rejected. Error code: " + str(reply.data))
            return False
        else: # Command was accepted
            return True

    def home(self, device):
        if device in self.devices:
            device = self.devices[device]
            reply = device.home()
            result = self.check_command_succeeded(reply)
            return result
        else:
            print("No device: ", device)
            return None

    def renumber(self):
        command = BinaryCommand(0,2,0)
        self.ser.write(command)
        time.sleep(4)
        reply = self.ser.read()
        result = self.check_command_succeeded(reply)
        return result

    def reset(self, device):
        if device in self.devices:
            device = self.devices[device]
            command = BinaryCommand(0,0,0)
            reply = device.send(command)
            result = self.check_command_succeeded(reply)
            return result
        else:
            print("No device: ", device)
            #return None

    def set_speed(self, device, speed):   # Speed is in degrees/second
        if device in self.devices:
            device = self.devices[device]
            action = 42
            res = 0.000115378 # microstep size.
            maxSpeed = .8

            if speed>maxSpeed:
                speed = maxSpeed #Maximum speed in degrees/second

            steps = round(speed*1./(res*1.6))
            command = BinaryCommand(0,action,steps)
            reply = device.send(command)
            result = self.check_command_succeeded(reply)
            if result:
                return(reply.data) #TODO: is this returnSpeed as below?
            else:
                return None


    def set_speed_knob(self, device, speed):   # Speed is in degrees/second
        if device in self.devices:
            device = self.devices[device]
            action = 111
            res = 0.000115378 # microstep size.
            maxSpeed = .8

            if speed>maxSpeed:
                speed = maxSpeed #Maximum speed in degrees/second

            steps = round(speed*1./(res*1.6))

            command = BinaryCommand(0,action,steps)
            reply = device.send(command)
            result = self.check_command_succeeded(reply)
            if result:
                return(reply.data) #see set_speed
            else:
                return None

    def set_speed_profile_knob(self, device, profile = 2):
        # profile: 1 = linear, 2 = quadratic, 3= cubic
        if device in self.devices:
            device = self.devices[device]
            action = 112 #changed from 111 in original (typo?)
            command = BinaryCommand(0,action,profile)
            reply = device.send(command)
            result = self.check_command_succeeded(reply)
            if result:
                return(reply.data) #see set_speed
            else:
                return None

    def set_acc(self, device, acc = 111):   # Default is 111
        if device in self.devices:
            device = self.devices[device]
            action - 43
            maxAcc = 111

            if acc>maxAcc:
                acc = maxAcc #Maximum speed in degrees/second

            command = BinaryCommand(0, action, acc)
            reply = device.send(command)
            result = self.check_command_succeeded(reply)
            if result:
                return(reply.data)
            else:
                return None


    def set_speed_all(self, speed):   # Speed is in degrees/second
        for device in self.devices:
            self.set_speed(device, speed)

    def set_acc_all(self, acc):   # Default is 111
        for device in self.devices:
            self.set_acc(device, acc)

    def get_speed(self,device):
        if device in self.devices:
            device = self.devices[device]
            action = 53
            cmd = 42
            command = BinaryCommand(0, action, cmd)
            reply = device.send(command)
            result = self.check_command_succeeded(reply)
            if result:
                return(reply.data)
            else:
                return None
        else:
            print("No device: ", device)
            return None

    def get_speed_knob(self,device):
        if device in self.devices:
            device = self.devices[device]
            action = 53
            cmd = 111
            command = BinaryCommand(0, action, command)
            reply = device.send(command)
            result = self.check_command_succeeded(reply)
            if result:
                return(reply.data)
            else:
                return None
        else:
            print("No device: ", device)
            return None

    def get_profile_knob(self,device):
        if device in self.devices:
            device = self.devices[device]
            action = 53
            cmd = 112
            command = BinaryCommand(0, action, cmd)
            reply = device.send(command)
            result = self.check_command_succeeded(reply)
            if result:
                return(reply.data)
            else:
                return None
        else:
            print("No device: ", device)
            return None

    def get_acc(self,device):
        if device in self.devices:
            device = self.devices[device]
            action = 53
            cmd = 43
            command = BinaryCommand(0, action, cmd)
            reply = device.send(command)
            result = self.check_command_succeeded(reply)
            if result:
                return(reply.data)
            else:
                return None
        else:
            print("No device: ", device)
            return None

    def get_speed_all(self):
        speeds = [-1] * len(self.devices)
        for i in range(len(self.devices)):
            print("Device number:", self.devices[i])
            speeds[i] = self.get_speed(self.devices[i])
        return(speeds)

    def get_acc_all(self):
        accs = [-1] * len(self.devices)
        for i in range(len(self.devices)):
            #print("Device number:", self.devices[i])
            accs[i] = self.get_acc(self.devices[i])
        return(accs)

    def move_relative(self, device, steps):
        if device in self.devices:
            device = self.devices[device]
            action = 21
            command = BinaryCommand(0, action, steps)
            try:
                reply = device.send(command)
                result = self.check_command_succeeded(reply)
                print('reply', type(reply.data))
                print('result', result)
                if result:
                    return(reply.data)
                else:
                    return(None)
            except:
                # return(self.get_position(device))
                return(-1)
        else:
            print("No device: ", device)
            return None

    def move_absolute(self, device, pos):
        if device in self.devices:
            device = self.devices[device]
            action = 20
            command = BinaryCommand(0, action, pos)
            reply = device.send(command)
            result = self.check_command_succeeded(reply)
            if result:
                return(reply.data)
            else:
                # return None
                return(self.get_position(device))
        else:
            print("No device: ", device)
            return None

    def get_position(self, device):
        if device in self.devices:
            device = self.devices[device]
            action = 60
            command = BinaryCommand(1, action)
            reply = device.send(command)
            print('checking position', device, reply)
            result = self.check_command_succeeded(reply)
            if result:
                return(reply.data)
            else:
                return None
            # return(position)
        else:
            print("No device: ", device)
            return None

    def get_firmware_cmd(self):
        return(8*self.potentiometer + 2*self.backlash + 4*self.stiction + (2**14 + 2**15)*self.LED)

    def backlash_correction(self, device, enabled = False):
        if device in self.devices:
            device = self.devices[device]
            if enabled:
                self.backlash= 1
            else:
                self.backlash = 0
            action = 40
            cmd = self.get_firmware_cmd()
            command = BinaryCommand(0, action, cmd)
            reply = self.devices[device].send(command)
            result = self.check_command_succeeded(reply)
            if result:
                return(reply.data)
            else:
                return None
            #msg = (device, 40, cmd)
            #msgCode = apply(self.struct.pack, msg)
            #self.ser.write(msgCode)
            time.sleep(.1)

    def stiction_correction(self, device, enabled = False):
        if device in self.devices:
            device = self.devices[device]
            if enabled:
                self.stiction = 1
            else:
                self.stiction = 0
            action = 40
            cmd = self.get_firmware_cmd()
            command = BinaryCommand(0, action, cmd)
            reply = device.send(command)
            result = self.check_command_succeeded(reply)
            if result:
                return(reply.data)
            else:
                return None
            #msg = (device, 40, cmd)
            #msgCode = apply(self.struct.pack, msg)
            #self.ser.write(msgCode)
            #returnMsgCoded = self.ser.read(self.packet_size)
            #returnMsg = self.struct.unpack(returnMsgCoded)
            time.sleep(0.1)

    def potentiometer_enabled(self, device, enabled = True):
        if device in self.devices:
            device = self.devices[device]
            #print(enabled)
            if enabled:
                self.potentiometer = 0
            else:
                self.potentiometer = 1
            action = 40
            cmd = self.get_firmware_cmd()
            command = BinaryCommand(0, action, cmd)
            reply = device.send(command)
            result = self.check_command_succeeded(reply)
            if result:
                return(reply.data)
            else:
                return None
            time.sleep(0.1)

#Only built-in on ASCII protocol which is not supported by T-series :(
    #def poll_until_idle(self, device):
    #   if device in self.devices:
    #        device.poll_until_idle()
    #    else:
    #        print ("No device, ", device)
    #        return None

#    def poll_until_idle(self, device):
#        if device in self.devices:
#            status = 1
#            while status > 0:
#                self.write(device, 55)
#                reply = self.read()
#                status = reply[2]
#                time.sleep(0.05)

    def write(self, device, action, cmd = 0):
        try:
            device = self.devices[device]
            command = BinaryCommand(0, action, cmd)
            reply = device.send(command)
            result = self.check_command_succeeded(reply)
            if result:
                return(reply.data)
            else:
                return None
        except:
            print("Failed to write: ", action, parameters)

    def read(self):
        try:
            reply = self.ser.read()
            result = self.check_command_succeeded(reply)
            if result:
                return(reply.data)
            else:
                return None
        except:
            return('')

    def LED_enabled(self, device, enabled = True):
        if device in self.devices:
            device = self.devices[device]
            if enabled:
                self.LED = 0
            else:
                self.LED = 1
            action = 40
            cmd = self.get_firmware_cmd()
            command = BinaryCommand(0, action, cmd)
            reply = device.send(command)
            result = self.check_command_succeeded(reply)
            if result:
                return(reply.data)
            else:
                return None
            time.sleep(0.1)

#
#port = 20 # COM port for the Zaber Motor.
#channels =  [1,2,3,4]
#zb = zaber_base(port, channels)
