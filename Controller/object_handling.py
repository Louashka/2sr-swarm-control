# Make grasping first:
# 1. Select the closest point on the object's contour
# 2. Approach the point (misth some pre margin) in a rigid mode
# 3. Grasp the object in a soft mode SM3

import sys
sys.path.append('D:/Robot 2SR/2sr-swarm-control')
from Model import global_var as gv, robot2sr as rsr, manipulandum
import motive_client, robot2sr_controller as rsr_ctrl, camera_optitrack_synchronizer as cos
import pandas as pd
import threading
from datetime import datetime
import numpy as np
import pandas as pd

mocap = motive_client.MocapReader()
rgb_camera = cos.Aligner()

agent: rsr.Robot = None
agent_controller = rsr_ctrl.Controller()

cheescake_contour = None
manip: manipulandum.Shape = None

def extractManipShape(path) -> list:
    contour_df = pd.read_csv(path)
    contour_r = contour_df['radius'].tolist()
    contour_theta = contour_df['phase_angle'].tolist()

    manip_contour_params = [contour_r, contour_theta]

    return manip_contour_params


def updateConfig():
    global agent, manip, markers
    # Get the current MAS and manipulandums configuration
    agent_config, manip_config, markers, msg = mocap.getAgentConfig(manip_exp_n=1)

    if agent_config and manip_config:
        if agent:
            agent.pose = [agent_config['x'], agent_config['y'], agent_config['theta']]
            agent.k1 = agent_config['k1']
            agent.k2= agent_config['k2']
        else:
            agent = rsr.Robot(agent_config['id'], agent_config['x'], agent_config['y'], agent_config['theta'], agent_config['k1'], agent_config['k2'])

        agent.head.pose = agent_config['head']
        agent.tail.pose = agent_config['tail']

        # print(f'Agent\'s pose: {agent.pose}')

        manip_pose = [manip_config['x'], manip_config['y'], manip_config['theta']]
        if manip:
            manip.pose = manip_pose
        else:
            manip = manipulandum.Shape(manip_config['id'], manip_pose, cheescake_contour)
        # print(f'Manip\'s pose: {manip.pose}')
    else:
        print('Agent and/or manipulandum is not detected! ' + msg)

def updateConfigLoop():
    while True:
        updateConfig()


if __name__ == "__main__":
    cheescake_contour = extractManipShape('Experiments/Data/Contours/cheescake_contour.csv')

    print('Start Motive streaming')
    mocap.startDataListener() 
    
    update_thread = threading.Thread(target=updateConfigLoop)
    update_thread.daemon = True  
    update_thread.start()

    while not agent or not manip:
        pass

    # ------------------------ Start a video ------------------------
    date_title = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rgb_camera.startVideo(date_title, task='object_handling')

    print("Waiting for the video to start...")
    while not rgb_camera.wait_video:
        pass

    print('Video started')
    print()
    # ---------------------------------------------------------------
    
    while True:
        rgb_camera.current_config = agent.config
        rgb_camera.markers = markers

    #     # Convert cheescake_contour to phase angles and radiuses with respect to manip.pose
    #     if rgb_camera.cheescake_contour and manip:
    #         manip_x, manip_y, manip_theta = manip.pose
    #         phase_angles = []
    #         radiuses = []
            
    #         for point in rgb_camera.cheescake_contour:
    #             # Translate the point relative to manip's position
    #             dx = point[0] - manip_x
    #             dy = point[1] - manip_y
                
    #             # Rotate the point to align with manip's orientation
    #             rotated_x = dx * np.cos(-manip_theta) - dy * np.sin(-manip_theta)
    #             rotated_y = dx * np.sin(-manip_theta) + dy * np.cos(-manip_theta)
                
    #             # Calculate phase angle and radius
    #             phase_angle = np.arctan2(rotated_y, rotated_x)
    #             radius = np.sqrt(rotated_x**2 + rotated_y**2)
                
    #             phase_angles.append(phase_angle)
    #             radiuses.append(radius)

    #         # Save phase angles and radiuses to CSV file
    #         csv_file_path = 'Experiments/Data/Contours/cheescake_contour.csv'
    #         # Create a DataFrame from the phase angles and radiuses
    #         df = pd.DataFrame({'phase_angle': phase_angles, 'radius': radiuses})

    #         # Save the DataFrame to a CSV file
    #         df.to_csv(csv_file_path, index=False)
    #         print(f"Cheesecake contour data saved to {csv_file_path}")

    #         break

            