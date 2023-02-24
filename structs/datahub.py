from numpy import array, zeros, ones, eye, trace, sin, cos, tan, arcsin, arccos, pi
from numpy.linalg import norm,inv

from lib.transformations    import *

class DataHub:



    def __init__(self, camera_info, params, inits ):
        '''
        # DataHub

        Public Data Storage Class 
        
        '''


        """ Images """
        
        ### Frame ###
        self.f_prev                         = array([])
        self.f_curr                         = array([])

        ### KeyFrame ###
        self.kf_prev                        = array([])
        self.kf_prev                        = array([])


        """ Camera Info """
        
        ### Camera Matrix ###
        self.K                              = camera_info["CAMERA_MTX"]
        self.K_inv                          = inv(self.K)
        
        ### Image Size ###
        self.img_size                       = camera_info["IMG_SIZE"]


        """ Parameters """
        
        ### Optimization Parameters ###
        self.param_stpcrt                   = params["PARAM_STOPPING_CRITERION"]
        self.param_maxitr                   = params["PARAM_MAX_ITERATION"]

        ### Translation Scale Parameter ###
        self.param_scale                    = params["PARAM_SCALE"]


        """ Optimizer Matrices """


        """ Points """
        self.points3D                       = []
        

        """ Poses """
        
        ### Initial Camera Pose ###
        self.T_W2B_init                     = T_W2B(Eular2R(inits["INIT_POSE"],dir=1),\
                                                    inits["INIT_LOCATION"])

        print(self.T_W2B_init)