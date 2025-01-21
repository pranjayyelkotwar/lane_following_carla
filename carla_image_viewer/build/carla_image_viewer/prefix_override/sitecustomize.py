import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/bitsauto/carla_experiments/lane_following_carla/carla_image_viewer/install/carla_image_viewer'
