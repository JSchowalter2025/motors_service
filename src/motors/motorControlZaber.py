import zmq

class MotorControllerZaber():
    def __init__(self,ip, port=54000, name = ''):
        self.context = zmq.Context()

        # Socket to talk to server
        print("Connecting to Zaber motor_server", ip, port)
        self.s = self.context.socket(zmq.REQ)
        self.s.connect("tcp://" + str(ip) + ':' +str(port))
        # msg = "Test"
        # reply = self._send(msg)
        #print(reply)

        self.id_dict = {}
        self.channels = []
        self.channelNames = []
        self.get_name()

    def _response(self):
        msg = self.s.recv()
        msg = msg.decode()
        if msg.split(" ")[0] == "Error:":
            print(msg)
            return -1
        return msg

    def _send(self,msg):
        msg = msg.encode()
        print('sending', msg)
        self.s.send(msg)
        reply = self._response()
        return reply

    def get_name(self):
        msg = self._send('zaber')
        motor_names = msg.split(',')[:-1]

        for pair in motor_names:
            pair = pair.split(':')
            ch = pair[0]
            chName = pair[1]
            self.id_dict[chName] = ch
            self.channels.append(ch)
            self.channelNames.append(chName)
        return(self.id_dict)

    def move_absolute(self, ch, pos):
        msg = ("zMoveAbs %s %s" %(ch, int(float(pos))))
        pos = self._send(msg)
        failed = pos == 'None'
        if failed == False:
        # if pos != '' or pos!=None:
            pos = int(float(pos))
        return(pos)

    def move_relative(self, ch, pos):
        msg = ("zMoveRel %s %s" %(ch, int(float(pos))))
        pos = self._send(msg)
        # print('pos', pos)
        failed = pos == 'None'
        # print('failed', failed)
        # if pos != '' or pos!=None or pos!='None':
        if failed == False:
            # print('inside conversion')
            pos = int(float(pos))
        else:
            pos = self.get_position(ch)
        return(pos)

    def move_all_to_position(self, pos):
        returnPos = []
        for i in range(len(self.channels)):
            ret = self.move_absolute(self.channels[i], pos[i])
            returnPos.append(ret)
        return returnPos

    def get_position(self, ch):
        msg = ("zGetPos %s" %(ch))
        pos = self._send(msg)
        if pos != '' or pos!=None:
            pos = int(float(pos))
        return(pos)

    def get_all_positions(self):
        positions = []
        for ch in self.channels:
            pos = (self.get_position(ch))
            positions.append(pos)
        return positions

    def set_knob_speed(self, ch, speed):
        msg = ("zSetKnobSpeed %s %s" %(ch, speed))
        pos = self._send(msg)
        if pos != '' or pos!=None:
            pos = int(float(pos))
        return(pos)

    def get_knob_speed(self, ch):
        msg = ("zGetKnobSpeed %s" %(ch))
        pos = self._send(msg)
        if pos != '' or pos!=None:
            pos = float(pos)
        return(pos)

    def set_speed(self, ch, speed):
        msg = ("zSetSpeed %s %s" %(ch, speed))
        pos = self._send(msg)
        if pos != '' or pos!=None:
            pos = int(float(pos))
        return(pos)

    def get_speed(self, ch):
        msg = ("zGetSpeed %s" %(ch))
        pos = self._send(msg)
        if pos != '' or pos!=None:
            pos = float(pos)
        return(pos)

    def home(self, ch):
        msg = ("zHome %s" %(ch))
        msgResp = self._send(msg)
        return(msgResp)

    def renumber(self):
        msg = ("zRenumber")
        msgResp = self._send(msg)
        return(msgResp)

    def potentiometer_enabled(self, ch, enabled = True):
        msg = ('zpotentiometer %s %s' %(ch, enabled) )
        msgResp = self._send(msg)
        return(msgResp)

    def potentiometer_all_enabled(self, enabled):
        ans = []
        for i in range(len(self.channels)):
            ret = self.potentiometer_enabled(self.channels[i], enabled)
            ans.append(ret)
        return ans

    def LED_enabled(self, ch, enabled = True):
        msg = ('zled %s %s' %(ch, enabled) )
        msgResp = self._send(msg)
        return(msgResp)


    def close(self):
        msg = ("zClose")
        msgResp = self._send(msg)
        return(msgResp)
