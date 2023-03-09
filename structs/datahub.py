from numpy import array, zeros, ones, eye, trace, sin, cos, tan, arcsin, arccos, pi
from numpy.linalg import norm,inv

from lib.transformations    import *

import rospy

class DataHub:



    def __init__(self, camera_info, ros_info, params, inits ):
        '''
        # DataHub

        Public Data Storage Class 
        
        '''


        """ Images """
        
        ### Frame ###
        self.frame_prev                     = array([])
        self.frame_curr                     = array([])

        ### KeyFrame ###
        self.kframe_prev                    = array([])
        self.kframe_prev                    = array([])


        """ Camera Info """
        
        ### Camera Matrix ###
        self.K                              = array(camera_info["CAMERA_MTX"])
        self.K_inv                          = inv(self.K)
        self.dist_coeff                     = array(camera_info["DIST_COEFF"])
        
        ### Image Size ###
        self.img_size                       = camera_info["IMG_SIZE"]


        """ ROS Info """

        ### Image Topic Name ###
        self.topic_name                     = ros_info["TOPIC_NAME"]


        """ Parameters """
        
        ### Feature Handling Paramters ###
        self.PARAM_mtchth                   = params["PARAM_MATCH_THRESHOLD"]
        self.PARAM_inlrth                   = params["PARAM_INLIER_THRESHOLD"]

        ### Optimization Parameters ###
        self.PARAM_stpcrt                   = params["PARAM_STOPPING_CRITERION"]
        self.PARAM_maxitr                   = params["PARAM_MAX_ITERATION"]

        ### Translation Scale Parameter ###
        self.PARAM_scale                    = params["PARAM_SCALE"]


        """ Optimizer Matrices """
        self.F_p                            = zeros((self.PARAM_mtchth*2,1),                  dtype=float)
        self.F_c                            = zeros((self.PARAM_mtchth*2,1),                  dtype=float)
        
        self.J                              = zeros((self.PARAM_mtchth*2,self.PARAM_mtchth+6),dtype=float)
        
        self.H                              = zeros((self.PARAM_mtchth+6,self.PARAM_mtchth+6),dtype=float)
        self.H_mod                          = zeros((self.PARAM_mtchth+6,self.PARAM_mtchth+6),dtype=float)
        
        self.x_p                            = zeros((self.PARAM_mtchth+6),                    dtype=float)
        self.x_c                            = zeros((self.PARAM_mtchth+6),                    dtype=float)


        """ Points """
        self.points3D                       = []


        """ Poses """
        
        ### Initial Camera Pose ###
        self.T_W2B_init                     = T_W2B(Eular2R(inits["INIT_POSE"],dir=1),\
                                                    inits["INIT_LOCATION"])


        """ Flags """

        self.FLAG_isFrameRecieved           = False