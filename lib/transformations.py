from numpy import array, zeros, ones, eye, trace, block, reshape, sin, cos, tan, arcsin, arccos, pi
from numpy.linalg import norm,inv



#########################################
## Fundamental Transformation Matrices ##
#########################################



def Eular2R(x, y, z, dir=0):
    '''
    ### Eular2R

    Calculate Rotation Matrix from Eular Angle

    Input   : Roll : rad, Pitch : rad, Yaw : rad, dir : bool 
    
    Output  : Rotation Body to World ( X_W = R @ X_B ) : if dir = 0

              Rotation world to Body ( X_B = R @ X_W ) : if dir = 1
    '''

    Rf_r = array([[ 1, cos(x), sin(x) ],
                  [ 0,-sin(x), cos(x) ],
                  [ 0,      0,     0  ]])

    Rf_p = array([[ cos(y), 0, -sin(y) ],
                  [      0, 1,      0  ],
                  [ sin(y), 0,  cos(y) ]])

    Rf_y = array([[ cos(z), sin(z), 0 ],
                  [-sin(z), cos(z), 0 ],
                  [      0,      0, 1  ]])


    if dir == 0: # Body to World

        R_B2W = Rf_r.T@Rf_p.T@Rf_y.T

        return R_B2W
        
    elif dir == 1: # world to Body

        R_W2B = Rf_y@Rf_p@Rf_r

        return R_W2B



def w2R(w):
    '''
    ### w2R
    
    Calculate Rotation Matrix (SO3) from Angular Vector (so3) via Rodrigues's Formula

    Input   : w = angle * axis  : double (3,1)
 
    Output  : Rotation Matrix   : double (3,3) 

    '''

    angle = norm(w)

    if angle == 0:  
        
        R = eye(3)
    
    else:

        n_w = w/angle

        n_w_hat = array([[   0, -n_w[2],  n_w[1]],
                         [ n_w[2],    0, -n_w[0]],
                         [-n_w[1],  n_w[0],    0]])

        R = (1-cos(angle))*(n_w_hat @ n_w_hat) + sin(angle)*n_w_hat + eye(3)

        R = R.T

    return R



def R2w(R):
    '''
    ### R2w
    
    Calculate Angular Vector (so3) from Rotation Matrix (SO3) via Log Mapping

    Input   : Rotation Matrix   : double (3,3) 
 
    Output  : w = angle * axis  : double (3,1)

    '''

    theta = arccos((trace(R)-1)/2)
    
    if theta == 0 or theta == pi:

        w = array([0,0,0])

    else:

        nw_hat = (1/(2*sin(theta)))*(R-R.T)

        w = array([nw_hat[2,1],nw_hat[0,2],nw_hat[1,0]])*theta

    return w



def T_W2B(R_W2B,t):
    '''
    ### T_W2B
    
    Form World to Body Transformation Matrix. ( X_B = T @ X_W )

    Input   : Rotation Matrix : double (3,3), Translation Vector : double (3,1)
 
    Output  : Transfromation Matrix : double (4,4)

    '''

    t = reshape(t,(3,1))

    T = block([[R_W2B, -R_W2B@t],
               [zeros((1,3)), 1]])

    return T



def T_B2W(R_W2B,t):
    '''
    ### T_B2W
    
    Form Body to World Transformation Matrix. ( X_W = T @ X_B )

    Input   : Rotation Matrix : double (3,3), Translation Vector : double (3,1)
 
    Output  : Transfromation Matrix : double (4,4)

    '''

    t = reshape(t,(3,1))

    T = block([[R_W2B.T, t],
               [zeros((1,3)), 1]])

    return T
    


def proj(cam, K, points3D):
    '''
    ### Image Plane Projection
    
    Project Body coordinates to Image Plane. ( x = K[I|0]X_B )

    Input   : Rotation Matrix : double (3,3), Translation Vector : double (3,1)
 
    Output  : Transfromation Matrix : double (4,4)

    '''

    P = K@cam.T_W2B[:3,:]

    points2D = P@points3D

    points2D = points2D/points2D[2]

    cam.points2D = points2D
