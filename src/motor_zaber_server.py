import datetime
import os
import yaml
from serial.tools import list_ports
from zmqhelper import ZMQServiceBase
from motors import zaber_base
from datetime import datetime


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
        self.config_dir  = './config/'
        self.config = self.load_config()
        
        self.n_workers = 1 # Zaber motors can only handle one command at a time
        self.time_start = datetime.now()
        
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
            loki_host = cParams['loki_host'],
            loki_port = cParams['loki_port'],
            redis_host = cParams['redis_host'],
            redis_port = cParams['redis_port'],
            service_name = cParams['service_name']
        )

        # these will be populated in setup()
        self._zb = []        
        self.setup()
        
        self.logger.info("")
        self.logger.info(f'{self.service_name} Zaber motor server Started at {self.time_start}')
        self.logger.info(f"Config: {self.config}")

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
                    self.logger.debug("zaber port reverting to yaml")
                    port = m['port']
                self.channels = m['channels']
                self.chNames = m['channelName']
                self._zb.append(zaber_base(port, self.channels))

    def handle_request(self, message: str) -> str:
        """
        Called in each worker thread.
        `message` is the raw bytes from the client.
        Return the raw bytes reply.
        """
        try:
            parts = message.split()
            cmd   = parts[0].lower()
            self.logger.debug(f"Received command: {message}")
            # ping
            if cmd == 'test':
                self.logger.debug("Ping command received")
                return "Connected"
            
            elif cmd == 'zaber':
                msgout = ''
                # self.logger.debug(f"Zaber command received, fetching channels {self.channels}, {self.chNames}")
                for i in range(len(self.channels)):
                    msgout += f"{self.channels[i]}:{self.chNames[i]}, "
                    # msgout += '%s:%s,' % (self.channels[i], self.chNames[i])
                self.logger.debug(f"Zaber channels: {msgout}")

            elif cmd == 'zmoveabs':
                msgout = self._zb[0].move_absolute(int(parts[1]), int(parts[2]))
                self.logger.debug(f'Absolute move command: {parts[1]} {parts[2]}')

            elif cmd == 'zmoverel':
                # self.logger.debug('relative', cmd, int(parts[2]))
                msgout = self._zb[0].move_relative(int(parts[1]), int(parts[2]))
                self.logger.debug(f'Relative move command: {parts[1]} {parts[2]}')

            elif cmd == 'zgetpos':
                msgout = self._zb[0].get_position(int(parts[1]))
                self.logger.debug(f"Get position command: {parts[1]}")

            elif cmd == 'zhome':
                self.logger.debug('zhoming')
                msgout = self._zb[0].home(int(parts[1]))
                self.logger.debug(f"Home command: {parts[1]}")

            elif cmd == 'zclose':
                msgout = self._zb[0].close()
                self.logger.debug("Closing Zaber connection")

            elif cmd == 'zrenumber':
                msgout = self._zb[0].renumber()
                # Need to reconnect to the device after renumbering
                self.logger.debug("Renumbering Zaber device")
                self.setup()
                self.logger.debug("Reconnected to Zaber device after renumbering")

            elif cmd == 'zsetknobspeed':
                msgout = self._zb[0].set_speed_knob(int(parts[1]), float(parts[2]))
                self.logger.debug(f"Set knob speed command: {parts[1]} {parts[2]}")

            elif cmd == 'zgetknobspeed':
                msgout = self._zb[0].get_speed_knob(int(parts[1]))
                self.logger.debug(f"Get knob speed command: {parts[1]}")

            elif cmd == 'zsetspeed':
                msgout = self._zb[0].set_speed(int(parts[1]), float(parts[2]))
                self.logger.debug(f"Set speed command: {parts[1]} {parts[2]}")
            elif cmd == 'zgetspeed':
                msgout = self._zb[0].get_speed(int(parts[1]))
                self.logger.debug(f"Get speed command: {parts[1]}")

            elif cmd == 'zpotentiometer':
                enabled = (parts[2] == 'True')
                msgout = self._zb[0].potentiometer_enabled(int(parts[1]), enabled)
                self.logger.debug(f"Set potentiometer enabled: {parts[1]} {enabled}")
            elif cmd == 'zled':
                enabled = (parts[2] == 'True')
                msgout = self._zb[0].LED_enabled(int(parts[1]), enabled)
                self.logger.debug(f"Set LED enabled: {parts[1]} {enabled}")

            else:
                msgout = "Invalid Command"
                self.logger.debug(f"Invalid command received: {cmd}")

            return str(msgout)
        
        except Exception as e:
            err = f"Error: {e}"
            self.logger.error(f"Exception in handle_request: {err}")
            return err

    @staticmethod
    def find_com(substring: str) -> str:
        """Helper to locate a Thorlabs LCC on a serial port."""
        for port in list_ports.comports():
            if substring.lower() in port[1].lower():
                return port.device
        raise RuntimeError(f"Could not find device matching {substring}")

if __name__ == '__main__':
    config_file = 'zaber_motors.yaml'
    service = MotorZaberZMQService(config_file)
    service.start()  # clean up sockets & context