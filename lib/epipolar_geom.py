from numpy        import array, diagonal, where, zeros, ones, eye, block, trace, sqrt, sin, cos, tan, arcsin, arccos, pi, deg2rad
from numpy.linalg import norm,inv,svd
import cv2



class EpipolarGeom:



    def __init__(self, DataHub):
        '''
        ## EpipolarGeom

        Estimates Camera Pose and 3D-Point Coordinates via Epipolar Geometry
        '''

        self.DataHub = DataHub


    
    def estimate_T(self, cam_prev, cam_curr):
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

        points2D_B1 = cam_prev.query_points2D
        points2D_B2 = cam_curr.train_points2D

        ### Find Essential Matrix ###
        E,mask = cv2.findEssentialMat(points2D_B1[:2].T, points2D_B2[:2].T, self.DataHub.K, cv2.RANSAC)

        ### Outliers Filtering ###
        inlier_idx = where(mask==1)[0]

        cam_prev.query_points2D = cam_prev.query_points2D[:,inlier_idx]
        cam_curr.train_points2D = cam_curr.train_points2D[:,inlier_idx]

        cam_prev.query_indices = cam_prev.query_indices[inlier_idx]
        cam_curr.train_indices = cam_curr.train_indices[inlier_idx]

        cam_prev.query_intensity = cam_prev.query_intensity[inlier_idx]
        cam_curr.train_intensity = cam_curr.train_intensity[inlier_idx]

        ### Recovering Pose with Filetered Point Pairs ###

        is_purerot = False

        if norm(diagonal(E)) < 0.01:
            ### Pure Rotation ###

            is_purerot = True

            points3D_B1 = self.DataHub.K_inv @ points2D_B1
            points3D_B2 = self.DataHub.K_inv @ points2D_B2

            R = points3D_B2 @ points3D_B1.T @ inv(points3D_B1 @ points3D_B1.T)

            T_B12B2 = block([[R,zeros((3,1))],[0,0,0,1]])

        else:
            ### General Motion ###

            _,R,t,_ = cv2.recoverPose(E,cam_prev.query_points2D[:2].T, cam_curr.train_points2D[:2].T,self.DataHub.K)


            T_B12B2 = block([[R,t],[0,0,0,1]])

        return T_B12B2, is_purerot

    
    def estimate_P_general(self, cam_prev, cam_curr, T_B12B2): 
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
        N = len(cam_prev.query_points2D[0])

        ### Prev Frame Projection Matrix ###
        P1 = self.DataHub.K@array([[1,0,0,0],
                                   [0,1,0,0],
                                   [0,0,1,0]])

        ### Curr Frame Projection Matrix ###
        P2 = self.DataHub.K@T_B12B2[:3,:]

        ### Matched 2D Points ###
        points2D_B1 = cam_prev.query_points2D
        points2D_B2 = cam_curr.train_points2D

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


    def intersecting_matches(self, cam_prev, cam_curr):

        prev_train_indices = cam_prev.train_indices
        prev_query_indices = cam_prev.query_indices
        curr_train_indices = cam_curr.train_indices

        # k-2 & k-1 matches ^ k-1 & k matches
        ints_train_matches_indices = zeros(len(prev_train_indices)+len(prev_query_indices), dtype=int)
        ints_query_matches_indices = zeros(len(prev_train_indices)+len(prev_query_indices), dtype=int)

        # k-2, k-1, k kp indices
        ints_prev_query_indices = zeros(len(prev_train_indices)+len(prev_query_indices), dtype=int)
        ints_curr_train_indices = zeros(len(prev_train_indices)+len(prev_query_indices), dtype=int)

        ints_num = 0

        for prev_train_match_idx, prev_train_idx in enumerate(prev_train_indices):

            prev_query_match_idx = where(prev_query_indices == prev_train_idx)[0]

            if len(prev_query_match_idx) != 0:
                # same keypoint detected in (k-2,k-1) & (k-1,k) matches 

                # matches[prev_matches_idx] and matches[curr_matches_idx] contains same keypoint of k-1 frame
                ints_prev_query_indices[ints_num] = prev_train_idx
                ints_curr_train_indices[ints_num] = curr_train_indices[prev_query_match_idx[0]]
                
                ints_train_matches_indices[ints_num]    = prev_train_match_idx
                ints_query_matches_indices[ints_num]    = prev_query_match_idx

                ints_num += 1

        ints_train_matches_indices  = ints_train_matches_indices[:ints_num]
        ints_query_matches_indices  = ints_query_matches_indices[:ints_num]
        ints_prev_query_indices     = ints_prev_query_indices[:ints_num]
        ints_curr_train_indices     = ints_curr_train_indices[:ints_num]

        return ints_train_matches_indices, ints_query_matches_indices


    def estimate_rel_scale(self, cam_prev, cam_curr, points3D_B1_curr):

        points3D_B1_prev = cam_prev.train_points3D

        train_match_indices, query_match_indices = self.intersecting_matches(cam_prev,cam_curr)

        ints_Z_B1_prev = points3D_B1_prev[2,train_match_indices]
        ints_Z_B1_curr = points3D_B1_curr[2,query_match_indices]

        rel_scale = sqrt((ints_Z_B1_prev.T @ ints_Z_B1_prev) / (ints_Z_B1_curr.T @ ints_Z_B1_curr))

        return rel_scale
    

    def estimate_P_purerot(self, cam_prev, cam_curr, T_B12B2):

        points3D_B1_prev = cam_prev.train_points3D

        train_match_indices, query_match_indices = self.intersecting_matches(cam_prev,cam_curr)

        cam_prev.query_indices      = cam_prev.query_indices[query_match_indices]
        cam_curr.train_indices      = cam_curr.train_indices[query_match_indices]

        cam_prev.query_intensity    = cam_prev.query_intensity[query_match_indices]
        cam_curr.train_intensity    = cam_curr.train_intensity[query_match_indices]

        if len(points3D_B1_prev) != 0:
    
            cam_prev.query_points3D     = cam_prev.train_points3D[:,train_match_indices]
            cam_curr.train_points3D     = T_B12B2 @ cam_prev.query_points3D

        else: pass

        cam_curr.T_W2B = T_B12B2@cam_prev.T_W2B
        cam_curr.T_B2W = inv(cam_curr.T_W2B)


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

        ### Estimate Transformation ###
        T_B12B2, is_purerot = self.estimate_T(cam_prev,cam_curr)

        
        ### Case #1 : Pure Rotation ###
        if is_purerot:

            self.estimate_P_purerot(cam_prev,cam_curr,T_B12B2)

        else:

            points3D_B1_prev = cam_prev.train_points3D
            points3D_B1_curr = self.estimate_P_general(cam_prev, cam_curr, T_B12B2)

            if len(points3D_B1_prev) == 0:

                abs_scale = self.DataHub.PARAM_scale

                T_B12B2[:3,3] = abs_scale * T_B12B2[:3,3]

                points3D_B1_curr_scaled = points3D_B1_curr
                points3D_B1_curr_scaled[:3,:] = abs_scale * points3D_B1_curr_scaled[:3,:]

                cam_prev.query_points3D = points3D_B1_curr_scaled
                cam_curr.train_points3D = T_B12B2 @ points3D_B1_curr_scaled
                
                cam_curr.T_W2B = T_B12B2@cam_prev.T_W2B
                cam_curr.T_B2W = inv(cam_curr.T_W2B)


            else:

                rel_scale = self.estimate_rel_scale(cam_prev, cam_curr, points3D_B1_curr)

                T_B12B2[:3,3] = rel_scale * T_B12B2[:3,3]

                points3D_B1_curr_scaled = points3D_B1_curr
                points3D_B1_curr_scaled[:3,:] = rel_scale * points3D_B1_curr_scaled[:3,:]

                cam_prev.query_points3D = points3D_B1_curr_scaled
                cam_curr.train_points3D = T_B12B2 @ points3D_B1_curr_scaled

                cam_curr.T_W2B = T_B12B2@cam_prev.T_W2B
                cam_curr.T_B2W = inv(cam_curr.T_W2B)
                

        return T_B12B2, cam_prev, cam_curr