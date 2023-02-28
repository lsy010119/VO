from numpy        import array, diagonal, where, zeros, ones, eye, block, trace, sin, cos, tan, arcsin, arccos, pi, deg2rad
from numpy.linalg import norm,inv,svd
import cv2



class EpipolarGeom:



    def __init__(self, DataHub):
        '''
        ## EpipolarGeom

        Estimates Camera Pose and 3D-Point Coordinates via Epipolar Geometry
        '''

        self.DataHub = DataHub


    
    def estimate_T(self, cam_prev, cam_curr, scale):
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

        points2D_B1 = cam_prev.points2D_with_curr
        points2D_B2 = cam_curr.points2D_with_prev

        ### Find Essential Matrix ###
        E,mask = cv2.findEssentialMat(points2D_B1[:2].T, points2D_B2[:2].T, self.DataHub.K, cv2.RANSAC)

        ### Outliers Filtering ###
        inlier_idx = where(mask==1)[0]

        cam_prev.points2D_with_curr = cam_prev.points2D_with_curr[:,inlier_idx]
        cam_curr.points2D_with_prev = cam_curr.points2D_with_prev[:,inlier_idx]

        cam_prev.tri_with_curr = cam_prev.tri_with_curr[inlier_idx]
        cam_curr.tri_with_prev = cam_curr.tri_with_prev[inlier_idx]

        cam_prev.intensity_with_curr = cam_prev.intensity_with_curr[inlier_idx]
        cam_curr.intensity_with_prev = cam_curr.intensity_with_prev[inlier_idx]

        ### Recovering Pose with Filetered Point Pairs ###

        is_purerot = False

        if norm(diagonal(E)) < 0.001:
            ### Pure Rotation ###

            is_purerot = True

            points3D_B1 = self.DataHub.K_inv @ points2D_B1
            points3D_B2 = self.DataHub.K_inv @ points2D_B2

            R = points3D_B2 @ points3D_B1.T @ inv(points3D_B1 @ points3D_B1.T)

            T_B12B2 = block([[R,zeros((3,1))],[0,0,0,1]])

        else:
            ### General Motion ###

            _,R,t,_ = cv2.recoverPose(E,cam_prev.points2D_with_curr[:2].T, cam_curr.points2D_with_prev[:2].T,self.DataHub.K)

            T_B12B2 = block([[R,scale*t],[0,0,0,1]])

        return T_B12B2, is_purerot

    
    def estimate_P(self, cam_prev, cam_curr, T_B12B2): 
        '''
        ### Estimate 3D-Points

        estimate the coordinate of 3D-points in body frame via triangulation

        Input

            points2D_B1 : double 3 X N
            points2D_B2 : double 3 X N
            T_B12B2     : double 4 X 4
            cam_prev    : Cam
            cam_curr    : Cam

        Output

            points3D_B1 : double 4 X N
        '''

        ### Number of Points to Triangulate ###
        N = len(cam_prev.points2D_with_curr[0])

        ### Prev Frame Projection Matrix ###
        P1 = self.DataHub.K@array([[1,0,0,0],
                                   [0,1,0,0],
                                   [0,0,1,0]])

        ### Curr Frame Projection Matrix ###
        P2 = self.DataHub.K@T_B12B2[:3,:]

        ### Matched 2D Points ###
        points2D_B1 = cam_prev.points2D_with_curr
        points2D_B2 = cam_curr.points2D_with_prev

        ### Init Memories ###
        points3D_B1 = zeros((4,N))

        for i in range(N):

            x1,y1 = points2D_B1[0,i],points2D_B1[1,i]
            x2,y2 = points2D_B2[0,i],points2D_B2[1,i]

            p1,p2,p3,p1_,p2_,p3_ = P1[0],P1[1],P1[2],P2[0],P2[1],P2[2]

            A = block([[ y1*p3 - p2  ],
                       [ p1 - x1*p3  ],
                       [ y2*p3_-p2_  ],
                       [ p1_ - x2*p3_]])
        
            _,_,VT = svd(A)

            point3D_B1 = VT[-1]/VT[-1,3]

            points3D_B1[:,i] = point3D_B1.T

        return points3D_B1


    def estimate_rel_scale(self, cam_prev, points3D_B1_curr):

        tri_with_prev = cam_prev.tri_with_prev
        tri_with_curr = cam_prev.tri_with_curr

        points3D_B1_prev = cam_prev.points3D_with_prev

        sum_Z_B1_prev_X_Z_B1_curr = 0
        sum_Z_B1_curr_X_Z_B1_curr = 0

        for i, prev_kpidx in enumerate(tri_with_prev):

            j = where(tri_with_curr == prev_kpidx)[0]

            if len(j) != 0:

                Z_B1_prev = points3D_B1_prev[2,i]    # keypoint with an index of prev_kpidx 
                Z_B1_curr = points3D_B1_curr[2,j[0]]

                sum_Z_B1_prev_X_Z_B1_curr += Z_B1_prev * Z_B1_curr
                sum_Z_B1_curr_X_Z_B1_curr += Z_B1_curr * Z_B1_curr
    

        rel_scale = sum_Z_B1_prev_X_Z_B1_curr/sum_Z_B1_curr_X_Z_B1_curr

        return rel_scale
    

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

        ### Translation Absolute Scale ###
        scale = self.DataHub.PARAM_scale

        ### Estimate Transformation ###
        T_B12B2, is_purerot = self.estimate_T(cam_prev,cam_curr,scale)

        ### Triangulate Points for Mapping and Calculating Relative Scale ###
        
        if is_purerot:

            points3D_B1_prev = cam_prev.points3D_with_prev

            if len(points3D_B1_prev) == 0:

                cam_curr.T_W2B = T_B12B2@cam_prev.T_W2B
                cam_curr.T_B2W = inv(cam_curr.T_W2B)

            else:

                cam_prev.points3D_with_curr = cam_prev.points3D_with_prev
                cam_curr.points3D_with_prev = T_B12B2 @ cam_prev.points3D_with_curr

                cam_curr.T_W2B = T_B12B2@cam_prev.T_W2B
                cam_curr.T_B2W = inv(cam_curr.T_W2B)


        else:

            points3D_B1_prev = cam_prev.points3D_with_prev
            points3D_B1_curr = self.estimate_P(cam_prev, cam_curr, T_B12B2)

            if len(points3D_B1_prev) == 0:

                cam_prev.points3D_with_curr = points3D_B1_curr
                cam_curr.points3D_with_prev = T_B12B2@points3D_B1_curr
                
                cam_curr.T_W2B = T_B12B2@cam_prev.T_W2B
                cam_curr.T_B2W = inv(cam_curr.T_W2B)


            else:

                rel_scale = self.estimate_rel_scale(cam_prev, points3D_B1_curr)

                T_B12B2[:3,3] = rel_scale * T_B12B2[:3,3]

                cam_prev.points3D_with_curr = rel_scale * points3D_B1_curr
                cam_curr.points3D_with_prev = T_B12B2@(rel_scale * points3D_B1_curr)

                cam_curr.T_W2B = T_B12B2@cam_prev.T_W2B
                cam_curr.T_B2W = inv(cam_curr.T_W2B)
                

        return T_B12B2, cam_prev, cam_curr