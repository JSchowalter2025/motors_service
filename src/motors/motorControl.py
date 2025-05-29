import zmq
import threading


class MotorController(threading.Thread):
    def __init__(self, ip, port):
        # need threading for asynchronous execution
        self.port = port
        threading.Thread.__init__(self)
        self.context = zmq.Context()
        # Socket to talk to server
        #print("Connecting to motor_server", ip, port)
        self.s = self.context.socket(zmq.REQ)
        self.s.connect("tcp://"+str(ip)+':'+str(port))
        #self.s.send(b"Test")
        #reply = self._response()
        #print(reply)

        self.id_dict = {}
        self.get_name()
        #print(self.id_dict)

    def _response(self):
        """Checks if there is an error message."""
        msg = self.s.recv()
        msg = msg.decode()
        # Check if msg is an error msg
        if msg.split(" ")[0] == "Error:":
            print(msg)
            return -1
        return msg

    def get_name(self):
        self.s.send("apt\n".encode())
        msg = self._response()
        if msg != -1:
            motor_names = msg.split(',')[:-1]
            self.id_dict = {}
            for count, n in enumerate(motor_names):
                self.id_dict[n] = count
            return(self.id_dict)

    def forward(self, name, distance):
        try:
            #print("%s forward %i" % (name, distance))
            outMessage = "for %s %i\n" % (distance, self.id_dict[name])
            self.s.send(outMessage.encode())
            reply = self._response()
            return reply
        except KeyError:
            return "Motor not connected"

    def backward(self, name, distance):
        try:
            #print("%s backward %i" % (name, distance))
            outMessage = "back %s %i\n" % (distance, self.id_dict[name])
            self.s.send(outMessage.encode())
            reply = self._response()
            return reply
        except KeyError:
            return "Motor not connected"

    def goto(self, name, pos):
        try:
            #print("Motor goto position %s" % pos)
            outMessage = "goto %s %i\n" % (pos, self.id_dict[name])
            self.s.send(outMessage.encode())
            reply = self._response()
            return reply

        except KeyError:
            return "Motor not connected"

    def home(self, name):
        try:
            print("Homing %s" % name)
            outMessage = "home %i \n" % self.id_dict[name]
            self.s.send(outMessage.encode())
            reply = self._response()
            return reply
        except KeyError:
            return "Motor not connected"

    def getAPos(self, name):
        try:
            outMessage = 'getapos %i \n' % self.id_dict[name]
            self.s.send(outMessage.encode())
            reply = self._response()
            pos = reply
            print("%s absolute position: %s" % (name, pos))
            return reply
        except KeyError:
            return "Motor not connected"

    def getPos(self, name):
        # print("getpos %s" %name)
        try:
            outMessage = 'getpos %i \n' % self.id_dict[name]
            self.s.send(outMessage.encode())
            reply = self._response()
            pos = reply
            #print("%s position: %s" % (name, pos))
            return reply
        except KeyError:
            return "Motor not connected"

    def getAllPos(self):
        positions = {}
        for motorName in self.id_dict.keys():
            pos = (self.getPos(motorName))
            pos = float(pos)
            positions[motorName] = pos
        return positions

    def close(self):
        self.s.close()
        print("Closed socket.")

if __name__ == "__main__":
    mc = MotorController(55000)
