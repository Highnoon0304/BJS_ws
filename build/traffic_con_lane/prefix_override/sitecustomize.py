import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/lee/Desktop/BJS_ws/install/traffic_con_lane'
