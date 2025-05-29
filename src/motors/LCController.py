#!/usr/bin/env python
# coding: utf-8


import serial as ser


class LCController(object):
    """Controller class for the thorlabs LCC25


    """

    def __init__(self, comport, timeout=5, baud=115200):
        super(LCController, self).__init__()
        self.serial_device = ser.Serial(comport)
        self.serial_device.baudrate = baud
        self.serial_device.timeout = timeout

    """low level methods"""

    def flush(self):
        temp = self.serial_device.timeout
        self.serial_device.timeout = 0.01
        while True:
            c = self.serial_device.read()
            if len(c) == 0:
                break
        self.serial_device.timeout = temp
        return 0

    def read(self):
        line = []
        while True:
            c = self.serial_device.read()
            if len(c) > 0:
                if c == b'>' and line[-1] == '\r':
                    return ''.join([li if li != '\r' else '\n' for li in line])
                line.append(c.decode())
            else:
                raise Exception("Device did not return response")

    def write(self, command):
        self.flush()
        self.serial_device.flushInput()
        self.serial_device.write((command + '\r').encode('ascii'))
        self.flush()
        return 0

    def query(self, command):
        self.flush()
        self.serial_device.flushInput()
        self.serial_device.write((command + '\r').encode('ascii'))

        return self.read()

    """High level methods"""

    def get_name(self):
        return "Device ID is: " + self.query('*idn?')

    def set_voltage(self, channel, voltage):
        if voltage > 0 and voltage < 25:
            self.write('volt{}={}'.format(channel,voltage))
            return 0
        else:
            raise Exception("Voltage must be between 0 and 25 volt")

    def get_voltage(self, channel):
        return self.query('volt{}?'.format(channel))[7:-1]

    def set_frequency(self, freq):
        if freq > 5 and freq < 150:
            self.write("freq={}".format(freq))
            return 0
        else:
            raise Exception("Frequency must be between 0 and 150")

    def get_frequency(self):
        return self.query("freq?")[6:-1]

    def set_out_mode(self, param):
        if param in [0, 1, 2]:
            self.write("mode={}".format(param))
            return 0
        else:
            raise Exception("argument to set_mode_out must be 0, 1 or 2")

    def get_out_mode(self):
        return self.query("mode?")[6:-1]

    def set_out_enable(self, en):
        if en in [0, 1]:
            self.write("enable={}".format(en))
            return 0
        else:
            raise Exception("argument to set_out_enable must be 0 or 1")

    def get_out_enable(self):
        return self.query('enable?')[8:-1]

    def set_ext_mod(self, extern):
        if extern in [0, 1]:
            self.write('extern={}'.format(extern))
            return 0
        else:
            raise Exception(
                "argument to set_ext_mod must be 0 for internal or 1 for external")

    def get_ext_mod(self):
        return self.query('extern?')[8:-1]

    def set_preset(self, preset):
        self.write('set={}'.format(preset))
        return 0

    def get_preset(self, preset):
        self.query('get={}'.format(preset))
        return 0

    def save_params(self):
        self.write('save')
        return 0

    def restore_def_params(self):
        self.write('default')
        return 0

    def set_test_dwell_time(self, time):
        """This is in ms"""
        self.write('dwell={}'.format(time+1))
        return 0

    def get_test_dwell_time(self):
        return self.query('dwell?')[7:-1] + 'ms'

    def set_test_inc(self, inc):
        """This is in volts"""
        self.write('increment={}'.format(inc))
        return 0

    def get_test_inc(self):
        return self.query('increment?')[11:-1] + 'V'

    def set_test_volt(self, arg,  volt):
        if arg == 'min':
            self.write('min={}'.format(volt))
        elif arg == 'max':
            self.write('max={}'.format(volt))
        return 0

    def get_test_volt(self):
        return "Min: " + self.query('min?')[5:-1] + " Max: " +\
            self.query('max?')[5:-1]

    def run_test_mode(self):
        self.write('test')
        return 0

    def remote_toggle(self, tog):
        if tog in [0, 1]:
            self.write('remote={}'.format(tog))
            return 0
        else:
            raise Exception("argto toggle must be 0 or 1")

    def help(self):
        return self.query('?')[2:-1]
