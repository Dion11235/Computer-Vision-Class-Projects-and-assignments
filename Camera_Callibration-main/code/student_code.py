import numpy as np


def calculate_projection_matrix(points_2d, points_3d):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:

                                                      [ M11      [ u1
                                                        M12        v1
                                                        M13        .
                                                        M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1        M22        .
      .  .  .  . .  .  .  .    .     .      .       *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]      M31        .
                                                        M32        un
                                                        M33 ]      vn ]

    Then you can solve this using least squares with np.linalg.lstsq() or SVD.
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs.

    Args:
    -   points_2d: A numpy array of shape (N, 2)
    -   points_2d: A numpy array of shape (N, 3)

    Returns:
    -   M: A numpy array of shape (3, 4) representing the projection matrix
    """

    # Placeholder M matrix. It leads to a high residual. Your total residual
    # should be less than 1.
    M = np.asarray([[0.1768, 0.7018, 0.7948, 0.4613],
                    [0.6750, 0.3152, 0.1136, 0.0480],
                    [0.1020, 0.1725, 0.7244, 0.9932]])

    ###########################################################################
    # TODO: YOUR PROJECTION MATRIX CALCULATION CODE HERE
    ###########################################################################

    #raise NotImplementedError('`calculate_projection_matrix` function in ' +
    #    '`student_code.py` needs to be implemented')
    
    A = np.zeros((2*points_3d.shape[0],11))
    k=0
    for i in range(points_3d.shape[0]):
        v1,v2 = points_3d[i].reshape(1,3), points_2d[i].reshape(1,2)
        homv1,homv2 = np.append(v1,1).reshape(1,4), np.append(v2,1).reshape(1,3)
        
        aughomv1 = np.concatenate((homv1,np.zeros((1,4))),axis=1)
        aughomv2 = np.concatenate((np.zeros((1,4)),homv1),axis=1)
        
        subpart1 = np.concatenate((aughomv1,aughomv2))
        subpart2 = ((-1)*v2.T) @ v1
        
        A[k:k+2] = np.concatenate((subpart1,subpart2),axis=1)
        k = k+2
    
    M_dummy = np.linalg.lstsq(A,points_2d.flatten())[0]
    M_dummy = np.append(M_dummy,1)
    M = M_dummy.reshape(3,4)
    M /= np.linalg.norm(M)

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return M

def calculate_camera_center(M):
    """
    Returns the camera center matrix for a given projection matrix.

    The center of the camera C can be found by:

        C = -Q^(-1)m4

    where your project matrix M = (Q | m4).

    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """

    # Placeholder camera center. In the visualization, you will see this camera
    # location is clearly incorrect, placing it in the center of the room where
    # it would not see all of the points.
    cc = np.asarray([1, 1, 1])

    ###########################################################################
    # TODO: YOUR CAMERA CENTER CALCULATION CODE HERE
    ###########################################################################

#     raise NotImplementedError('`calculate_camera_center` function in ' +
#         '`student_code.py` needs to be implemented')
    
    Q = M[:,0:3]
    m4 = M[:,3]
    
    cc = -np.linalg.inv(Q) @ m4

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return cc

def estimate_fundamental_matrix(points_a, points_b):
    """
    Calculates the fundamental matrix. Try to implement this function as
    efficiently as possible. It will be called repeatedly in part 3.

    You must normalize your coordinates through linear transformations as
    described on the project webpage before you compute the fundamental
    matrix.

    Args:
    -   points_a: A numpy array of shape (N, 2) representing the 2D points in
                  image A
    -   points_b: A numpy array of shape (N, 2) representing the 2D points in
                  image B

    Returns:
    -   F: A numpy array of shape (3, 3) representing the fundamental matrix
    """

    # Placeholder fundamental matrix
    F = np.asarray([[0, 0, -0.0004],
                    [0, 0, 0.0032],
                    [0, -0.0044, 0.1034]])

    ###########################################################################
    # TODO: YOUR FUNDAMENTAL MATRIX ESTIMATION CODE HERE
    ###########################################################################

#     raise NotImplementedError('`estimate_fundamental_matrix` function in ' +
#         '`student_code.py` needs to be implemented')
    
    hom_points_a = np.concatenate((points_a,np.ones((points_a.shape[0],1))),axis=1)
    hom_points_b = np.concatenate((points_b,np.ones((points_b.shape[0],1))),axis=1)
    
    A = np.zeros((points_a.shape[0],9))
    for i in range(points_a.shape[0]):
        u = points_a[i]
        
        diag1 = hom_points_b[i][0] * np.eye(3)
        diag2 = hom_points_b[i][1] * np.eye(3)
        diag3 = np.eye(3)
        
        B2 = np.concatenate((diag1,diag2,diag3),axis=1)
        B1 = hom_points_a[i].reshape((1,3))
        
        B = B1 @ B2
        
        A[i] = B
    
    #print(A.shape)
    U,S,V = np.linalg.svd(A)
    #print(U.shape,S.shape,V.shape)
    F = V[V.shape[1]-1,:].reshape(3,3)
    U1,S1,V1 = np.linalg.svd(F)
    S1[len(S1)-1]=0
    
    S1 = np.diagflat(S1)
    #print(S1)
    F = U1 @ S1 @ V1
    

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return F

def ransac_fundamental_matrix(matches_a, matches_b):
    """
    Find the best fundamental matrix using RANSAC on potentially matching
    points. Your RANSAC loop should contain a call to
    estimate_fundamental_matrix() which you wrote in part 2.

    If you are trying to produce an uncluttered visualization of epipolar
    lines, you may want to return no more than 100 points for either left or
    right images.

    Args:
    -   matches_a: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image A
    -   matches_b: A numpy array of shape (N, 2) representing the coordinates
                   of possibly matching points from image B
    Each row is a correspondence (e.g. row 42 of matches_a is a point that
    corresponds to row 42 of matches_b)

    Returns:
    -   best_F: A numpy array of shape (3, 3) representing the best fundamental
                matrix estimation
    -   inliers_a: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image A that are inliers with
                   respect to best_F
    -   inliers_b: A numpy array of shape (M, 2) representing the subset of
                   corresponding points from image B that are inliers with
                   respect to best_F
    """

    # Placeholder values
    best_F = estimate_fundamental_matrix(matches_a[:10, :], matches_b[:10, :])
    inliers_a = matches_a[:100, :]
    inliers_b = matches_b[:100, :]

    ###########################################################################
    # TODO: YOUR RANSAC CODE HERE
    ###########################################################################

    num_iterator = 30000
    threshold = 0.001
    best_F = np.zeros((3, 3))
    max_inlier = 0
    num_rand_samples = 8
    
    hom_matches_a = np.concatenate((matches_a,np.ones((matches_a.shape[0],1))),axis=1)
    hom_matches_b = np.concatenate((matches_b,np.ones((matches_b.shape[0],1))),axis=1)
    
    A = np.zeros((matches_a.shape[0],9))
    for k in range(matches_a.shape[0]):
        u = matches_a[k]
        
        diag1 = hom_matches_b[k][0] * np.eye(3)
        diag2 = hom_matches_b[k][1] * np.eye(3)
        diag3 = np.eye(3)
        
        B2 = np.concatenate((diag1,diag2,diag3),axis=1)
        B1 = hom_matches_a[k].reshape((1,3))
        
        B = B1 @ B2
        
        A[k] = B

    for i in range(num_iterator):
        index_rand = np.random.randint(matches_a.shape[0], size=num_rand_samples)
        F_matrix = estimate_fundamental_matrix(matches_a[index_rand, :], matches_b[index_rand, :])
        err = np.abs(np.matmul(A, F_matrix.reshape((-1))))
        current_inlier = np.sum(err <= threshold)
        if current_inlier > max_inlier:
            best_F = F_matrix.copy()
            max_inlier = current_inlier

    err = np.abs(np.matmul(A, best_F.reshape((-1))))
    index = np.argsort(err)

    inliers_a = matches_a[index[:max_inlier]]
    inliers_b = matches_b[index[:max_inlier]]


    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return best_F, inliers_a, inliers_b