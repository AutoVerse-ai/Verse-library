import rosbag
import pandas as pd
from datetime import datetime
import os
import numpy as np 
import cv2
import sys

def rosbag_to_csv(bag_file, output_dir):
    # Open the bag file
    bag = rosbag.Bag(bag_file)
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get the list of topics
    topics = bag.get_type_and_topic_info()[1].keys()
    i = 0
    for topic in topics:
        messages = []
        for topic, msg, t in bag.read_messages(topics=[topic]):
            if topic == '/jaxguam/pose':
                msg_dict = {}
                msg_dict['timestamp'] = t.to_sec()
                tx = msg.pose.position.x
                ty = msg.pose.position.y
                tz = msg.pose.position.z
                rx = msg.pose.orientation.x
                ry = msg.pose.orientation.y
                rz = msg.pose.orientation.z
                rw = msg.pose.orientation.w
                msg_dict['tx'] = tx
                msg_dict['ty'] = ty
                msg_dict['tz'] = tz
                msg_dict['rx'] = rx
                msg_dict['ry'] = ry
                msg_dict['rz'] = rz
                msg_dict['rw'] = rw
                messages.append(msg_dict)
            elif topic == '/minihawk/pose':
                msg_dict = {}
                msg_dict['timestamp'] = t.to_sec()
                tx = msg.pose.position.x
                ty = msg.pose.position.y
                tz = msg.pose.position.z
                rx = msg.pose.orientation.x
                ry = msg.pose.orientation.y
                rz = msg.pose.orientation.z
                rw = msg.pose.orientation.w
                msg_dict['tx'] = tx
                msg_dict['ty'] = ty
                msg_dict['tz'] = tz
                msg_dict['rx'] = rx
                msg_dict['ry'] = ry
                msg_dict['rz'] = rz
                msg_dict['rw'] = rw
                messages.append(msg_dict)
            elif topic == "/jaxguam/velocity":
                msg_dict = {}
                msg_dict['timestamp'] = t.to_sec()
                tvx = msg.linear.x
                tvy = msg.linear.y 
                tvz = msg.linear.z 
                rvx = msg.angular.x 
                rvy = msg.angular.y
                rvz = msg.angular.z 
                msg_dict['tvx'] = tvx
                msg_dict['tvy'] = tvy
                msg_dict['tvz'] = tvz
                msg_dict['rvx'] = rvx
                msg_dict['rvy'] = rvy
                msg_dict['rvz'] = rvz
                messages.append(msg_dict)
            elif topic == "/target/pose":
                msg_dict = {}
                msg_dict['timestamp'] = t.to_sec()
                msg_dict['tvx'] = msg.linear.x
                msg_dict['tvy'] = msg.linear.y
                msg_dict['tvz'] = msg.linear.z
                msg_dict['rvx'] = msg.angular.x
                msg_dict['rvy'] = msg.angular.y
                msg_dict['rvz'] = msg.angular.z
                messages.append(msg_dict)
            else:
                continue
                # Convert ROS message to dictionary
                msg_dict = {}
                for slot in msg.__slots__:
                    msg_dict[slot] = getattr(msg, slot)
                msg_dict['timestamp'] = t.to_sec()
                messages.append(msg_dict)
        
        # Convert list of dictionaries to pandas DataFrame
        df = pd.DataFrame(messages)
        
        # Write DataFrame to CSV
        csv_file = os.path.join(output_dir, topic.replace('/', '_') + '.csv')
        df.to_csv(csv_file, index=False)
    
    bag.close()

# Usage

script_dir = os.path.dirname(os.path.realpath(__file__))

# fn_list = [
#     "20241205213753",
#     "20241205214208",
#     # "20241204154010",
#     # "20241204154352",
#     # "20241204154828",
# ]

if __name__ == "__main__":
    output_folder = sys.argv[1]
    output_folder = os.path.join(script_dir, output_folder)
    counter = 0
    for i, name in enumerate(os.listdir(output_folder)):
        if name.startswith('2024'):
            fn = os.path.join(output_folder, name, './tmp/recorded_topics.bag')
            print(fn)
            rosbag_to_csv(fn, os.path.join(output_folder, f'./extracted_{counter}/'))
            counter += 1
    # for i in range(2):
    #     fn = os.path.join(output_folder, fn_list[i], './tmp/recorded_topics.bag')
    #     # img_dir = os.path.join(script_dir, './pi_images/')
    #     # if not os.path.exists(img_dir):
    #     #     os.mkdir(img_dir)
    #     rosbag_to_csv(fn, os.path.join(output_folder, f'./extracted_{i}/'))
