from numpy import array, zeros, ones, eye, trace, sin, cos, tan, arcsin, arccos, pi
from numpy.linalg import norm,inv



class DataHub:



    def __init__(self, K, img_size, params, T_W2B_init):
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
        self.K                              = K
        self.K_inv                          = inv(K)
        
        ### Image Size ###
        self.img_size                       = img_size


        """ Parameters """
        
        ### Optimization Parameters ###
        self.param_stpcrt                   = params[0]
        self.param_maxitr                   = params[1]

        ### Translation Scale Parameter ###
        self.param_scale                    = params[2]


        """ Optimizer Matrices """


        """ Points """
        self.points3D                       = []
        

        """ Poses """
        
        ### Initial Camera Pose ###
        self.T_W2B_init                     = T_W2B_init
