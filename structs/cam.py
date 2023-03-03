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

        ### Maatched Feature Indices ###
        self.train_indices = array([])          # Indices of KeyPoints matched with prev frame
        self.query_indices = array([])          # Indices of KeyPoints matched with next frame

        ### Point Coordinates ###
        self.train_points2D = array([])         # Image Coordinate KeyPoints matched with prev frame
        self.query_points2D = array([])         # Image Coordinate KeyPoints matched with next frame

        self.train_points3D = array([])         # Body Coordinate 3D Points triangulated with prev frame
        self.query_points3D = array([])         # Body Coordinate 3D Points triangulated with next frame

        ### Intensity ###
        self.train_intensity = array([])        # Intensity of keypoints matched with prev frame
        self.query_intensity = array([])        # Intensity of keypoints matched with curr frame
