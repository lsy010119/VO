from gpg import Data
from numpy        import array, zeros, ones, eye, block, trace, diag, arange, reciprocal, deg2rad, sin, cos, tan, arcsin, arccos, pi
from numpy.linalg import norm,inv,svd
import cv2

from lib.transformations import *
import matplotlib.pyplot as plt


class BundleAdjustment:



    def __init__(self,DataHub):
        
        self.DataHub = DataHub

        self.N = DataHub.PARAM_mtchth

        self.x_p        = zeros((self.N+6),                      dtype=float)
        self.x_c        = zeros((self.N+6),                      dtype=float)

        self.del_x      = zeros((self.N+6,1))

        self.F_p        = zeros((self.N*2,1),                    dtype=float)
        self.F_c        = zeros((self.N*2,1),                    dtype=float)
        
        self.J          = zeros((self.N*2,self.N+6),     dtype=float)
        self.J_d        = zeros((self.N*2,self.N),     dtype=float)
        self.J_t        = zeros((self.N*2,6),                     dtype=float)

        self.H          = zeros((self.N+6,self.N+6),             dtype=float)
        
        
    def _dfdd1(self, p1, P2, R):

        K, K_inv = self.DataHub.K, self.DataHub.K_inv

        fx,fy = K[0,0], K[1,1]

        X2,Y2,Z2 = P2[0], P2[1], P2[2]

        dfdp2 = eye(2)

        dp2dP2 = array([[fx/Z2, 0, - X2*fx/(Z2**2)],
                        [0, fy/Z2, - Y2*fy/(Z2**2)]])

        dP2dd1 = R @ K_inv @ p1

        dfdd1 = dfdp2 @ dp2dP2 @ dP2dd1

        return dfdd1


    def _dfdt(self, P1, P2):

        K = self.DataHub.K

        fx,fy = K[0,0], K[1,1]

        X1,Y1,Z1 = P1[0], P1[1], P1[2]

        X2,Y2,Z2 = P2[0], P2[1], P2[2]

        dfdp2 = eye(2)

        dp2dP2 = array([[fx/Z2, 0, - X2*fx/(Z2**2)],
                        [0, fy/Z2, - Y2*fy/(Z2**2)]])

        dP2dt = array([[0, Z1, -Y1, 1, 0, 0],
                       [-Z1, 0, X1, 0, 1, 0],
                       [Y1, -X1, 0, 0, 0, 1]])

        dfdt = dfdp2 @ dp2dP2 @ dP2dt

        return dfdt


    def _update_J(self, p1, P1, P2, R):

        for i in range(self.N):

            p1i = p1[:,i]
            P1i = P1[:,i]
            P2i = P2[:,i]
            
            self.J[2*i:2*i+2, i] = self._dfdd1(p1i,P2i,R)
            self.J[2*i:2*i+2,-6:] = self._dfdt(P1i,P2i)         


    def _update_F(self, p2, P2):

        K, K_inv = self.DataHub.K, self.DataHub.K_inv

        self.F_p = self.F_c.copy()

        Z2 = P2[2]

        p2_hat = K@P2/Z2

        for i in range(self.N):

            p2i_hat = p2_hat[:2,i]
            p2i     = p2[:2,i]

            self.F_c[2*i:2*i+2,0] = p2i_hat - p2i


    def _update_H(self, lam):

        self.H = self.J.T @ self.J

        self.H = (self.H + lam*eye(self.N+6))
        # self.H = (self.H + lam*diag(diag(self.H)))


    def _update_delx(self):

        # H_dd = self.H[:-6,:-6]
        # H_dt = self.H[:-6,-6:]
        # H_td = self.H[-6:,:-6]
        # H_tt = self.H[-6:,-6:]
        # H_dd_inv = diag(reciprocal(diag(H_dd)))

        # b = self.J.T@self.F_c
        # b_d = b[:-6]
        # b_t = b[-6:]

        # self.del_x[-6:] = inv(-H_td@H_dd_inv@H_dt + H_tt)@(b_t - H_td@H_dd_inv@b_d)
        # self.del_x[:-6] = H_dd_inv@(b_d - H_dt@self.del_x[-6:])

        self.del_x  = - inv(self.H)@self.J.T@self.F_c


    def _optimize(self,p1,p2,x_init, img):


        # fig = plt.figure().add_subplot(1,1,1)

        # cost = []


        lam = 0.1
        stop = self.DataHub.PARAM_stpcrt
        N_iter = self.DataHub.PARAM_maxitr

        K, K_inv = self.DataHub.K, self.DataHub.K_inv
        
        R = w2R(x_init[-6:-3])
        t = reshape(x_init[-3:], (3,1))

        Z1 = x_init[:self.N]
        P1 = K_inv@(p1*Z1)
        P2 = R @ P1 + t

        #########
        # fig.cla()

        # Z2 = P2[2]

        # p2_hat = K@P2/Z2

        # fig.imshow(img,cmap='Greys_r')

        # for i in range(self.N):

        #     p1i = p1[:2,i]
        #     p2i = p2[:2,i]
        #     p2i_hat = p2_hat[:2,i]

        #     fig.scatter(p1i[0], p1i[1], color="red")
        #     fig.scatter(p2i[0], p2i[1], color="green")
        #     fig.scatter(p2i_hat[0], p2i_hat[1], color="yellow")

        # fig.scatter(p1i[0], p1i[1], color="red", label=r"$p^{(k)}$")
        # fig.scatter(p2i[0], p2i[1], color="green", label=r"$p^{(k+1)}$")
        # fig.scatter(p2i_hat[0], p2i_hat[1], color="yellow", label=r"$\hat{p}^{(k+1)}$")

        # fig.set_xlim(0,640)
        # fig.set_ylim(480,0)

        # fig.legend()

        #########

        self.x_c = x_init
        self.x_p = self.x_c.copy()

        self._update_F(p2,P2)
        self._update_J(p1, P1, P2, R)
        self._update_H(lam)
        self._update_delx()

        self.x_c = self.x_c + self.del_x[:,0]

        # cost.append(norm(self.F_c))

        for _ in range(N_iter-1):

            R = w2R(self.x_c[-6:-3])
            t = reshape(self.x_c[-3:], (3,1))

            Z1 = self.x_c[:self.N]
            P1 = K_inv@(p1*Z1)
            P2 = R @ P1 + t

            self.x_p = self.x_c.copy()


            self._update_F(p2,P2)
            self._update_J(p1, P1, P2, R)
            self._update_H(lam)
            self._update_delx()

            self.x_c = self.x_c + self.del_x[:,0]

            norm_F_1 = norm(self.F_p)
            norm_F_2 = norm(self.F_c)

            #########
            # cost.append(norm(self.F_c))
            
            # fig.cla()

            # Z2 = P2[2]

            # p2_hat = K@P2/Z2

            # fig.imshow(img,cmap='Greys_r')

            # for i in range(self.N):

            #     p1i = p1[:2,i]
            #     p2i = p2[:2,i]
            #     p2i_hat = p2_hat[:2,i]

            #     fig.scatter(p1i[0], p1i[1], color="red")
            #     fig.scatter(p2i[0], p2i[1], color="green")
            #     fig.scatter(p2i_hat[0], p2i_hat[1], color="yellow")

            # fig.scatter(p1i[0], p1i[1], color="red", label=r"$p^{(k)}$")
            # fig.scatter(p2i[0], p2i[1], color="green", label=r"$p^{(k+1)}$")
            # fig.scatter(p2i_hat[0], p2i_hat[1], color="yellow", label=r"$\hat{p}^{(k+1)}$")

            # fig.set_xlim(0,640)
            # fig.set_ylim(480,0)

            # fig.legend()

            # plt.pause(0.01)

            #########

            if norm_F_2 < norm_F_1:

                if (norm_F_1 - norm_F_2)/norm_F_1 <= stop:

                    print("converged")
                    break

                lam *= 0.5

            else:

                self.x_p = self.x_c
                self.x_c = self.x_c + self.del_x[:,0]

                lam *= 1.2

        # plt.plot(arange(len(cost)),cost)

        # plt.show()



    def run(self, cam_prev, cam_curr, T_B12B2):


        query_points2D = cam_prev.query_points2D
        train_points2D = cam_curr.train_points2D

        query_points3D = cam_prev.query_points3D

        w,t = R2w(T_B12B2[:3,:3]), T_B12B2[:3,3]

        x_init = zeros((self.N + 6))

        x_init[:self.N] = query_points3D[2]
        x_init[-6:-3] = w
        x_init[-3:] = t

        self._optimize(query_points2D,train_points2D, x_init, cam_prev.img)

