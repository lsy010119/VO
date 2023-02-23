from numpy        import array, deg2rad, zeros, ones, eye, block, trace, sin, cos, tan, arcsin, arccos, pi
from numpy.linalg import norm,inv,svd
import cv2



class EpipolarGeom:



    def __init__(self, DataHub):
        '''
        ## EpipolarGeom

        Estimates Camera Pose and 3D-Point Coordinates via Epipolar Geometry
        '''

        self.DataHub = DataHub


    
    def estimate_T(self, points2D_B1, points2D_B2, scale):
        '''
        ### Estimate Transformation

        estimate the transformation matrix via epipolar geometry with given 2D-point pairs
        
        Input

            points2D_B1 : double 3 X N
            points2D_B2 : double 3 X N
            scale       : double

        Output

            T_B12B2     : double 4 X 4
        '''

        E,_ = cv2.findEssentialMat(points2D_B1[:2].T, points2D_B2[:2].T, self.DataHub.K, cv2.RANSAC)

        _,R,t,_ = cv2.recoverPose(E,points2D_B1[:2].T, points2D_B2[:2].T,self.DataHub.K)

        T_B12B2 = block([[R,scale*t],[0,0,0,1]])

        return T_B12B2


    
    def estimate_P(self, points2D_B1, points2D_B2, T_B12B2): 
        '''
        ### Estimate 3D-Points

        estimate the coordinate of 3D-points in body frame via triangulation

        Input

            points2D_B1 : double 3 X N
            points2D_B2 : double 3 X N
            T_B12B2     : double 4 X 4

        Output

            points3D_B1 : double 4 X N
        '''

        N = len(points2D_B1)

        P1 = self.DataHub.K@array( [[1,0,0,0],
                                    [0,1,0,0],
                                    [0,0,1,0]])

        P2 = self.DataHub.K@T_B12B2[:3,:]

        points3D_B1 = zeros((4,N))

        for i in range(N):

            x1,y1 = points2D_B1[0,i],points2D_B1[1,i]
            x2,y2 = points2D_B2[0,i],points2D_B2[1,i]

            p1,p2,p3,p1_,p2_,p3_ = P1[0],P1[1],P1[2],P2[0],P2[1],P2[2]

            A = block([[y1*p3-p2],
                       [p1-x1*p3],
                       [y2*p3_-p2_],
                       [p1_-x2*p3_]])
        
            _,_,VT = svd(A)

            point3D_B1 = VT[-1]/VT[-1,3]

            points3D_B1[:,i] = point3D_B1.T

        return points3D_B1


    
    def track_pose(self, cam_prev, cam_curr):
        '''
        ### Pose Tracking

        tracks the pose of camera via epipolar gemoetry

        Input

            cam_prev    : Cam
            cam_curr    : Cam

        Output

            T_B12B2     : double 4 X 4
            cam_prev    : Cam
            cam_curr    : Cam

        '''

        scale = self.DataHub.param_scale

        points2D_B1 = cam_prev.points2D
        points2D_B2 = cam_curr.points2D

        T_B12B2 = self.estimate_T(points2D_B1,points2D_B2,scale)

        points3D_B1_prev = cam_prev.points3D
        points3D_B1_curr = self.estimate_P(points2D_B1,points2D_B2,T_B12B2)


        if len(points3D_B1_prev) == 0:

            cam_curr.points3D = T_B12B2@points3D_B1_curr

            cam_curr.T_W2B = T_B12B2@cam_prev.T_W2B
            cam_curr.T_B2W = inv(cam_curr.T_W2B)


        else:

            Z_B1_prev = points3D_B1_prev[2]
            Z_B1_curr = points3D_B1_curr[2]
            
            rel_scale = (Z_B1_prev.T@Z_B1_curr)/(Z_B1_curr.T@Z_B1_curr)

            T_B12B2[:3,3] = rel_scale * T_B12B2[:3,3]

            cam_curr.points3D = T_B12B2@points3D_B1_prev

            cam_curr.T_W2B = T_B12B2@cam_prev.T_W2B
            cam_curr.T_B2W = inv(cam_curr.T_W2B)
            
        return T_B12B2, cam_prev, cam_curr