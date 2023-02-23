from numpy import array, deg2rad, zeros, ones, eye, trace, sin, cos, tan, arcsin, arccos, pi
from numpy.linalg import norm,inv

from lib.transformations        import *
from lib.datahub                import DataHub
from lib.cam                    import Cam
from lib.epipolar_geom          import EpipolarGeom
from visualizer                 import Visualizer



class Main:

    def __init__(self, K, img_size, params, T_W2B_init, points3D=zeros((4,0))):

        self.points3D = points3D

        self.DataHub    = DataHub(K, img_size, params, T_W2B_init)
        self.visualizer = Visualizer(self.DataHub)
        self.epgeom     = EpipolarGeom(self.DataHub)

    def sequancial_transform(self, w_set, t_set):

        K = len(w_set[0])

        Cams = []

        T_W2B1 = eye(4)


        prev = 0

        for i in range(K):

            w = w_set[:,i]        
            t = t_set[:,i]

            R_B12B2 = w2R(w)        
            t_B12B2 = t        

            T_B12B2 = T_W2B(R_B12B2,t_B12B2)

            print(f"==={norm(T_B12B2[:3,3])}===")
            print(f"={norm(T_B12B2[:3,3])/prev}=")
            prev = norm(T_B12B2[:3,3])


            T_W2B2 = T_B12B2@T_W2B1

            Cami = Cam()

            Cami.T_W2B = T_W2B2
            Cami.T_B2W = inv(T_W2B2)

            Cams.append(Cami)

            T_W2B1 = T_W2B2

        return Cams


    def project_points(self, Cams,points3D):

        for cam in Cams:

            cam.points3D = cam.T_W2B@points3D
    
            proj(cam,self.DataHub.K,points3D)


    def run(self):

        ### Ground Truths ###

        w_set = array([ [deg2rad(-90),0,0,0,0],
                        [0,deg2rad(-20),deg2rad(-20),deg2rad(-20),deg2rad(-20)],
                        [0,0,0,0,0]])

        t_set = array([ [2,2,3,1,2],
                        [0,1,-1,1,0],
                        [3,1,1,1,1]])

        Cams = self.sequancial_transform(w_set,t_set)
        self.project_points(Cams,self.points3D)


        ### Estimation ###

        cam_prev = Cam()
        cam_curr = Cam()

        cam_prev.points2D = Cams[0].points2D
        cam_curr.points2D = Cams[1].points2D

        cam_prev.T_W2B    = Cams[0].T_W2B
        cam_prev.T_B2W    = Cams[0].T_B2W

        scale        = 2.4494897427831783
        # scale        = 1

        for k in range(4):


            T_B12B2, cam_prev_, cam_curr_ = self.epgeom.track_pose(cam_prev, cam_curr)

            cam_prev = cam_curr_

            cam_curr = Cam()
            cam_curr.points2D = Cams[k+1].points2D


            self.visualizer.viz_pose(cam_prev.T_B2W,color=0)
            self.visualizer.viz_pose(Cams[k].T_B2W,color=1)

        self.visualizer.run(self.points3D)


        

if __name__ == "__main__":

    K        = array([[ 300, 0,  200 ], 
                    [ 0, 300,  100 ],
                    [ 0, 0,    1 ]])    
    img_size =(3,2)
    params = [1,2,2.4494897427831783]
    T_W2B_init = eye(4)

    points3D = array([[-1,  0,  1,  1,  0, -1, -1,  0,  1 ],
                      [ 6,  6,  6,  7,  7,  7,  8,  8,  8 ],
                      [ 1,  2,  3,  1,  2,  3,  1,  2,  3 ],
                      [ 1,  1,  1,  1,  1,  1,  1,  1,  1 ]])

    VO = Main(K, img_size, params, T_W2B_init,points3D)
    VO.run()