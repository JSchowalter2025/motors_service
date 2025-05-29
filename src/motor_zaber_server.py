import datetime
import os
import yaml
from serial.tools import list_ports
from zmqhelper import ZMQServiceBase
from motors import zaber_base

class MotorZaberZMQService(ZMQServiceBase):
    """
    MotorZaberZMQService does:
      - binds a ROUTER socket for clients (tcp://*:54000)
      - binds a DEALER socket for workers (inproc://workers)
      - proxies ROUTER <> DEALER
    """
    def __init__(self, config_file):
        # global logger
        self.config_file = config_file
        self.config_dir  = 'config'
        self.config = self.load_config()
        
        self.n_workers = 1 # Zaber motors can only handle one command at a time
        self.time_start = datetime.now()

        # these will be populated in setup()
        self._zb = []
        self.logger.info("")
        self.logger.info(f'Zaber motor server Started at {self.time_start}')
        self.logger.info(f"Config: {self.config}")
        
        self.setup()
        
        cParams = self.config['config_setup']
        if 'redis_host' not in cParams or cParams['register_redis'] is False:
            cParams['redis_host'] = None 
        if 'loki_host' not in cParams:
            cParams['loki_host'] = None
        if 'redis_port' not in cParams:
            cParams['redis_port'] = None
        if 'loki_port' not in cParams:
            cParams['loki_port'] = None
        
        super().__init__(rep_port = cParams['req_port'], 
            n_workers= self.n_workers,
            http_port = cParams['http_port'],
            service_name = cParams['name'],
            loki_host = cParams['loki_host'],
            loki_port = cParams['loki_port'],
            redis_host = cParams['redis_host'],
            redis_port = cParams['redis_port']
        )

    def load_config(self):
        """Load YAML config file."""
        cfg_path = os.path.join(self.config_dir, self.config_file)
        with open(cfg_path, 'r') as f:
            return yaml.safe_load(f)
        
    def setup(self):
        """Load YAML config & initialize motor"""
        config = self.load_config()
        motor_dicts = config['motors']
        for m in motor_dicts:
            if m['type'].lower() == 'zaber':
                try:
                    port = self.find_com('ftdi ft232r usb')[-1]
                except:
                    print("zaber port reverting to yaml")
                    port = m['port']
                channels = m['channels']
                chNames = m['channelName']
                self._zb.append(zaber_base(port, channels))

    def handle_request(self, message: str) -> str:
        """
        Called in each worker thread.
        `message` is the raw bytes from the client.
        Return the raw bytes reply.
        """
        try:
            parts = message.split()
            cmd   = parts[0].lower()

            # ping
            if cmd == 'test':
                return b"Connected"

            elif cmd[0] == 'zmoveabs':
                msgout = self._zb[0].move_absolute(int(cmd[1]), int(cmd[2]))

            elif cmd[0] == 'zmoverel':
                # print('relative', cmd, int(cmd[2]))
                msgout = self._zb[0].move_relative(int(cmd[1]), int(cmd[2]))
                print('finished relative move ', msgout)

            elif cmd[0] == 'zgetpos':
                msgout = self._zb[0].get_position(int(cmd[1]))
                print(msgout, "gautam")

            elif cmd[0] == 'zhome':
                print('zhoming')
                msgout = self._zb[0].home(int(cmd[1]))

            elif cmd[0] == 'zclose':
                msgout = self._zb[0].close()

            elif cmd[0] == 'zrenumber':
                msgout = self._zb[0].renumber()
                # Need to reconnect to the device after renumbering
                self.setup()

            elif cmd[0] == 'zsetknobspeed':
                msgout = self._zb[0].set_speed_knob(int(cmd[1]), float(cmd[2]))

            elif cmd[0] == 'zgetknobspeed':
                msgout = self._zb[0].get_speed_knob(int(cmd[1]))

            elif cmd[0] == 'zsetspeed':
                msgout = self._zb[0].set_speed(int(cmd[1]), float(cmd[2]))

            elif cmd[0] == 'zgetspeed':
                msgout = self._zb[0].get_speed(int(cmd[1]))

            elif cmd[0] == 'zpotentiometer':
                enabled = (cmd[2].decode() == 'True')
                msgout = self._zb[0].potentiometer_enabled(int(cmd[1]), enabled)
            elif cmd[0] == 'zled':
                enabled = (cmd[2].decode() == 'True')
                msgout = self._zb[0].LED_enabled(int(cmd[1]), enabled)

            else:
                msgout = "Invalid Command"


        except Exception as e:
            err = f"Error: {e}"
            print(err)
            return err.encode()

    @staticmethod
    def find_com(substring: str) -> str:
        """Helper to locate a Thorlabs LCC on a serial port."""
        for port in list_ports.comports():
            if substring.lower() in port[1].lower():
                return port.device
        raise RuntimeError(f"Could not find device matching {substring}")

if __name__ == '__main__':
    service = MotorZaberZMQService()
    service.run()    # block until keyboard interrupt
    service.shutdown()  # clean up sockets & context

