import sys
from Motive.nat_net_client import NatNetClient
import mas_controller

class MocapReader:
    def __init__(self) -> None:
        self.__isRunning = False
        self.__data = None

    @property
    def isRunning(self) -> bool:
        return self.__isRunning
    
    @isRunning.setter
    def isRunning(self, value) -> None:
        self.__isRunning = value

    @property
    def data(self) -> object:
        return self.__data

    def __receiveData(self, value) -> None:
        self.__data = value

    def __parseArgs(self, arg_list, args_dict) -> dict:
        arg_list_len = len(arg_list)
        if arg_list_len > 1:
            args_dict["serverAddress"] = arg_list[1]
            if arg_list_len > 2:
                args_dict["clientAddress"] = arg_list[2]
            if arg_list_len > 3:
                if len(arg_list[3]):
                    args_dict["use_multicast"] = True
                    if arg_list[3][0].upper() == "U":
                        args_dict["use_multicast"] = False
        return args_dict
    
    def startDataListener(self) -> None:
        options_dict = {}
        options_dict["clientAddress"] = "127.0.0.1"
        options_dict["serverAddress"] = "127.0.0.1"
        options_dict["use_multicast"] = True

        # Start Motive streaning
        options_dict = self.__parseArgs(sys.argv, options_dict)

        streaming_client = NatNetClient()
        streaming_client.set_client_address(options_dict["clientAddress"])
        streaming_client.set_server_address(options_dict["serverAddress"])
        streaming_client.set_use_multicast(options_dict["use_multicast"])

        streaming_client.mocap_data_listener = self.__receiveData

        self.isRunning = streaming_client.run()

    def getConfig(self) -> list:
        mas = None
        manipulandums = None

        markers, rigid_bodies = self.__unpackData()
        if len(markers) == 0 or len(rigid_bodies) == 0:
            raise Exception("No agents are found!")
    

        return mas, manipulandums
    
    def __unpackData(self) -> list:
        if self.data is None:
            raise Exception('No data received from Motive!')
        
        rigid_body_data = self.data.rigid_body_data
        labeled_marker_data = self.data.labeled_marker_data

        labeled_marker_list = labeled_marker_data.labeled_marker_list
        rigid_body_list = rigid_body_data.rigid_body_list

        markers = {}
        rigid_bodies = {}

        for marker in labeled_marker_list:
            model_id, marker_id = [int(i) for i in marker.get_id()] 
            marker = {'model_id': model_id, 'marker_id': marker_id,
                        'marker_x': marker.pos[0], 'marker_y': marker.pos[1], 'marker_z': marker.pos[2]}
            markers[str(model_id) + '.' + str(marker_id)] = marker

        for rigid_body in rigid_body_list:
            rigid_body = {'id': int(rigid_body.id_num), 'x': rigid_body.pos[0], 'y': rigid_body.pos[1], 'z': rigid_body.pos[2], 'a': rigid_body.rot[0],
                            'b': rigid_body.rot[1], 'c': rigid_body.rot[2], 'd': rigid_body.rot[3]}
            rigid_bodies[int(rigid_body['id'])] = rigid_body    

        return markers, rigid_bodies



