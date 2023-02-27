from numpy import array, zeros, ones, eye
from numpy.linalg import inv



class Cam:



    def __init__(self, img = array([]) , T_W2B = eye(4)):
        '''
        # Camera
        
        Members
            
            img         : int h X w
            T_W2B       : double 4 X 4  
            T_W2B       : double 4 X 4
            points3D    : Point3D[N]  
            points2D    : Point2D[N]  

        Methods

            FuckMySelf  : No returns you piece of shit
        '''

        ### Image ###
        self.img = img

        ### Camera Pose ###
        self.T_W2B = T_W2B                      # X_B = T_W2B @ T_W
        self.T_B2W = inv(T_W2B)                 # X_W = T_B2W @ T_B

        ### Extracted Features ###
        self.keypoints = array([])              # Keypoints
        self.desc      = array([])              # SIFT Descriptors

        ### Point Coordinates ###
        self.points3D = array([])               # Body Coordinate 3D Points
        
        self.points3D_with_prev = array([])     # Body Coordinate 3D Points triangulated with prev frame
        self.points3D_with_curr = array([])     # Body Coordinate 3D Points triangulated with next frame

        self.tri_with_prev = array([])          # Indices of Points triangulated with prev frame
        self.tri_with_next = array([])          # Indices of Points triangulated with next frame

        ### Intensity ###
        self.intensity_with_prev = array([])    # Intensity of keypoints matched with prev frame
        self.intensity_with_curr = array([])    # Intensity of keypoints matched with curr frame
