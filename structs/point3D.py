from numpy import array, deg2rad, zeros, ones, eye, trace, sin, cos, tan, arcsin, arccos, pi
from numpy.linalg import norm,inv



class Point3D:



    def __init__(self, id, X_H = array([[0],[0],[0],[1]]), converged = False):
        '''
        ## Point3D

        Point class containing the world coordinate info

        Members
            
            id          : int
            loc         : double 4 X 1
            converged   : bool

        '''

        self.id         = id        # Point ID
        self.X_H        = X_H       # Homogeneous Coordinate
        self.converged  = converged # Is converged in optimization process



#############################
## Point Handler Functions ##
#############################



def Points3D2Array( points3D_lst ):
    '''
    ## Points3D2Array
    
    Input
        
        points3D_lst    : Point3D [N]

    Output

        points3D_arr    : double 4 X N
        pointsID_arr    : int N

    '''

    N = len(points3D_lst)

    points3D_arr = zeros((4,N))
    pointsID_arr = zeros(N)

    for idx,point3D in enumerate(points3D_lst):

        points3D_arr[:,idx]     = point3D.X_H
        pointsID_arr[idx]       = point3D.id

    return points3D_arr, pointsID_arr



def Array2Points3D( points3D_arr, pointsID_arr ):
    '''
    ## Array2Points3D
    
    Input
        
        points3D_arr    : double 4 X N
        pointsID_arr    : int N

    Output

        points3D_lst    : Point3D [N]

    '''

    N = len(pointsID_arr)

    points3D_lst = [Point3D()]*N

    for idx, pointID in enumerate(pointsID_arr):

        points3D_lst[idx].id    = pointID
        points3D_lst[idx].X_H   = points3D_arr[:,idx]

    return points3D_lst