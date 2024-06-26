import serial
import numpy as np
from typing import List
from Model import global_var, robot2sr
import sys

port_name = "COM3"

class Controller:
    def __init__(self) -> None:
        self.serial_port = serial.Serial(port_name, 115200)
        self.velocity_coef = 2
        self.t = 0

    def motionPlanner(self, agent: robot2sr.Robot, target: np.ndarray ) -> tuple[List[float], List[float]]:
        v = [0] * 5
        s = [0] * 2

        error = np.linalg.norm(agent.config - target)

        # INVERSE KINEMATICS

        dist = target - agent.config
        dist_angle = self.min_angle_distance(agent.theta, target[2])
        dist[2] = dist_angle

        q_tilda = self.velocity_coef * dist * global_var.DT
        v:np.ndarray = np.matmul(np.linalg.pinv(agent.jacobian), q_tilda)

        v = v.tolist()

        self.t += global_var.DT

        return v, s

    def move(self, agent: robot2sr.Robot, v, s) -> list:
        head_wheels, head_wheels_original = self.__calcWheelsCoords(agent.pose, agent.head.pose)
        tail_wheels, tail_wheels_original = self.__calcWheelsCoords(agent.pose, agent.tail.pose, lu_type='tail')

        wheels = head_wheels + tail_wheels

        omega = self.__calcWheelsVelocities(wheels, v, s)
        omega = np.array([5, 0, -5, 0])
        print(omega)
        
        commands = omega.tolist() + s + [agent.id]
        # print(commands)

        self.__sendCommands(commands)

        return head_wheels_original + tail_wheels_original

    def stop(self, agent: robot2sr.Robot) -> None:   
        commands = [0, 0, 0, 0] + agent.stiffness + [agent.id]
        self.__sendCommands(commands)

    def min_angle_distance(self, initial_angle, target_angle):
        # Calculate the clockwise and counterclockwise distances
        clockwise_distance = (target_angle - initial_angle) % (2 * np.pi)
        counterclockwise_distance = (initial_angle - target_angle) % (2 * np.pi)
        
        # Return the minimum of the two distances
        return min(clockwise_distance, counterclockwise_distance)

    def __calcWheelsCoords(self, agent_pose: list, lu_pose: list, lu_type='head'):
        if lu_type == 'head':
            w1_0 = 2 * np.array([[-0.0275], [0]])
            w2_0 = 2 * np.array([[0.0105], [-0.0275]])
        elif lu_type == 'tail':
            w1_0 = 2 * np.array([[0.0275], [0]])
            w2_0 = 2 * np.array([[-0.0105], [-0.027]])

        R = np.array([[np.cos(lu_pose[2]), -np.sin(lu_pose[2])],
                    [np.sin(lu_pose[2]), np.cos(lu_pose[2])]])
        w1 = np.matmul(R, w1_0).T[0] + lu_pose[:2]
        w2 = np.matmul(R, w2_0).T[0] + lu_pose[:2]

        w = self.__wheelsToBodyFrame(agent_pose, [w1, w2], lu_pose[-1], lu_type)

        return w, [w1, w2]

    def __wheelsToBodyFrame(self, agent_pose: list, w: list, lu_theta: float, lu_type='head'):
        R_ob = np.array([[np.cos(agent_pose[2]), -np.sin(agent_pose[2])],
                        [np.sin(agent_pose[2]), np.cos(agent_pose[2])]])
        
        T_ob = np.block([[R_ob, np.array([agent_pose[:2]]).T], [np.zeros((1,2)), 1]])
        T_bo = np.linalg.inv(T_ob)

        if lu_type == 'head':
            offset = 0
        elif lu_type == 'tail':
            offset = 2

        for i in range(2):
            w_b0 = [w[i][0], w[i][1], 1]
            w[i] = np.matmul(T_bo, w_b0).T[:-1]
            w[i] = np.append(w[i], (lu_theta - agent_pose[2]) % (2 * np.pi) + global_var.BETA[i+offset])

        return w


    def __calcWheelsVelocities(self, wheels, v, s) -> np.ndarray:
    
        flag_soft = int(s[0] or s[1])
        flag_rigid = int(not (s[0] or s[1]))

        V_ = np.zeros((4, 5))
        for i in range(4):
            w = wheels[i]
            tau = w[0] * np.sin(w[-1]) - w[1] * np.cos(w[-1])
            V_[i, :] = [flag_soft * int(i == 0 or i == 1), -flag_soft * int(i == 2 or i == 3), 
                        flag_rigid * np.cos(w[-1]), flag_rigid * np.sin(w[-1]), flag_rigid * tau]

        V = 1 / global_var.WHEEL_R * V_
        omega = np.matmul(V, v).round(3)

        # omega[np.abs(omega) < 2] = 0

        return omega

    def __sendCommands(self, commands):
        msg = "s"

        for command in commands:
            msg += str(command) + '\n'

        self.serial_port.write(msg.encode())