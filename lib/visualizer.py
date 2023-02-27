from numpy                  import array, zeros, ones, eye, trace, reshape, sin, cos, tan, arcsin, arccos, pi
from numpy.linalg           import norm,inv

from structs.cam            import Cam
from lib.transformations    import *

from mpl_toolkits.mplot3d   import Axes3D
import matplotlib.pyplot    as plt



class Visualizer:

    def __init__(self, DataHub):
        
        self.world = plt.figure(figsize=(10,10)).add_subplot(1,1,1,projection="3d")

        self.world.set_xlim(-5,5)
        self.world.set_ylim(0,10)
        self.world.set_zlim(0,10)

        self.DataHub = DataHub


    def viz_world(self):

        x_w = array([[1],[0],[0],[1]],dtype=float)
        y_w = array([[0],[1],[0],[1]],dtype=float)
        z_w = array([[0],[0],[1],[1]],dtype=float)

        self.world.quiver(0,0,0,x_w[0,0],x_w[1,0],x_w[2,0],arrow_length_ratio=0.1,color='black')
        self.world.quiver(0,0,0,y_w[0,0],y_w[1,0],y_w[2,0],arrow_length_ratio=0.1,color='black')
        self.world.quiver(0,0,0,z_w[0,0],z_w[1,0],z_w[2,0],arrow_length_ratio=0.1,color='black')
        self.world.text(0,0,0,"O",style="normal")
        self.world.text(1,0,0,r"$X_w$",style="normal")
        self.world.text(0,1,0,r"$Y_w$",style="normal")
        self.world.text(0,0,1,r"$Z_w$",style="normal")



    def viz_pose(self, T_B2W, color = 1):

        x_w = array([[1],[0],[0],[1]],dtype=float)
        y_w = array([[0],[1],[0],[1]],dtype=float)
        z_w = array([[0],[0],[1],[1]],dtype=float)

        x_c = T_B2W@x_w - reshape(T_B2W[:,3],(4,1))
        y_c = T_B2W@y_w - reshape(T_B2W[:,3],(4,1))
        z_c = T_B2W@z_w - reshape(T_B2W[:,3],(4,1))

        if color == 1:
            self.world.quiver(T_B2W[0,3],T_B2W[1,3],T_B2W[2,3],x_c[0,0],x_c[1,0],x_c[2,0],color='red')
            self.world.quiver(T_B2W[0,3],T_B2W[1,3],T_B2W[2,3],y_c[0,0],y_c[1,0],y_c[2,0],color='green')
            self.world.quiver(T_B2W[0,3],T_B2W[1,3],T_B2W[2,3],z_c[0,0],z_c[1,0],z_c[2,0],color='blue')

        else:

            self.world.quiver(T_B2W[0,3],T_B2W[1,3],T_B2W[2,3],x_c[0,0],x_c[1,0],x_c[2,0],color='black')
            self.world.quiver(T_B2W[0,3],T_B2W[1,3],T_B2W[2,3],y_c[0,0],y_c[1,0],y_c[2,0],color='black')
            self.world.quiver(T_B2W[0,3],T_B2W[1,3],T_B2W[2,3],z_c[0,0],z_c[1,0],z_c[2,0],color='black')



    def viz_poses(self):

        for i,cam in enumerate(self.DataHub.Cams):

            T_B2W = cam.T_B2W

            self.viz_pose(T_B2W)


    def viz_points3D(self, points3D):

        for i in range(len(points3D[0])):

            point3D = points3D[:3,i]

            self.world.scatter(point3D[0],point3D[1],point3D[2],color='black')


    def viz_points(self,points, intensity = []):

        if len(intensity) != 0:

            self.world.scatter(points[0],points[1],points[2],c=intensity,cmap="Greys")

        else:

            self.world.scatter(points[0],points[1],points[2],c='red')


    def viz_imgs(self):

        imgs = plt.figure(figsize=(10,10))

        for i,cam in enumerate(self.DataHub.Cams):

            img = imgs.add_subplot(len(self.DataHub.Cams), 2, i+1)
            
            img.set_title(f"Cam{i}")
            img.set_xlim(0,2*self.DataHub.K[0,2])
            img.set_ylim(2*self.DataHub.K[1,2],0)
            img.set_xticks([])
            img.set_yticks([])
            
            img.scatter(cam.points2D[0], cam.points2D[1])
            

    def viz_ray(self, cam, color='red'):
        
        T_B2W = cam.T_B2W

        infpoints_B = 100*self.DataHub.K_inv@cam.points2D

        for i in range(self.DataHub.N):

            infpoints_W = T_B2W[:3,:3]@infpoints_B[:,i] + T_B2W[:3,3]

            self.world.plot([T_B2W[0,3],infpoints_W[0]],[T_B2W[1,3],infpoints_W[1]],[T_B2W[2,3],infpoints_W[2]],\
                color=color,linestyle='-',linewidth=0.2)


    def viz_estimated(self,T_W2B,cam_gt):

        cam_estimated = Cam()
        cam_estimated.T_W2B    = T_W2B
        cam_estimated.T_B2W    = inv(T_W2B)
        cam_estimated.points2D = cam_gt.points2D
        
        self.viz_pose(cam_estimated.T_B2W)

        self.world.text(cam_estimated.T_B2W[0,3],cam_estimated.T_B2W[1,3],cam_estimated.T_B2W[2,3],"*",style="normal")
        # self.viz_ray(cam_estimated,'blue')

        return cam_estimated


    def viz_trajec(self, cam_hist):

        traj = zeros((3,len(cam_hist)))

        for i,cam in enumerate(cam_hist):

            self.viz_pose(cam.T_B2W)
            
            traj[:,i] = cam.T_B2W[:3,3]

            self.viz_points(cam.T_B2W@cam.points3D_with_prev, cam.intensity_with_prev)

        self.world.plot(traj[0],traj[1],traj[2],'r-')


    def run(self):

        self.viz_world()
        # self.viz_points3D(points3D)
        # self.viz_imgs()
        # self.viz_poses()

        self.world.axis("off")
        plt.show()