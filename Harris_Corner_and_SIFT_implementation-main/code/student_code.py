import numpy as np
import cv2 # You must not use cv2.cornerHarris()
# You must not add any other library


### If you need additional helper methods, add those. 
### Write details description of those

"""
  Returns the harris corners,  image derivative in X direction,  and 
  image derivative in Y direction.
  Args
  - image: numpy nd-array of dim (m, n, c)
  - window_size: The shaps of the windows for harris corner is (window_size, wind)
  - alpha: used in calculating corner response function R
  - threshold: For accepting any point as a corner, the R value must be 
   greater then threshold * maximum R value. 
  - nms_size = non maximum suppression window size is (nms_size, nms_size) 
    around the corner
  Returns 
  - corners: the list of detected corners
  - Ix: image derivative in X direction
  - Iy: image derivative in Y direction

"""
def harris_corners(image, window_size=5, sigma=1, alpha=0.04, threshold=1e-2,
                  nms_size=10):
    
    image = cv2.GaussianBlur(image,(5,5),3)

    Ix = cv2.Sobel(image,cv2.CV_64F,dx=1,dy=0)
    Iy = cv2.Sobel(image,cv2.CV_64F,dx=0,dy=1)
    
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy
    
    ker = cv2.getGaussianKernel(window_size,sigma)
    ker = ker @ ker.T
    sigIx2 = cv2.filter2D(Ix2,-1,ker)
    sigIy2 = cv2.filter2D(Iy2,-1,ker)
    sigIxy = cv2.filter2D(Ixy,-1,ker)
    
    det = (sigIx2 * sigIy2) - (sigIxy * sigIxy)
    trace = sigIx2 + sigIy2
    R = det - alpha*(trace*trace)
    
    #thresholding
    R[R<threshold*np.max(R)] = 0
    
    #Non-Maximum Suppression
    for i in range(R.shape[0]-nms_size):
        for j in range(R.shape[1]-nms_size):
            dummy = R[i:i+nms_size,j:j+nms_size]
            dummy[dummy != np.max(dummy)] = 0
    
    R1 = R.copy()
    R1[R1!=0]=1
    
    corners = R
    return corners, R1, Ix, Iy

"""
  Creates key points form harris corners and returns the list of keypoints. 
  You must use cv2.KeyPoint() method. 
  Args
  - corners:  list of Normalized corners.  
  - Ix: image derivative in X direction
  - Iy: image derivative in Y direction
  - threshold: only select corners whose R value is greater than threshold
  
  Returns 
  - keypoints: list of cv2.KeyPoint
  
  Notes:
  You must use cv2.KeyPoint() method. You should also pass 
  angle of gradient at the corner. You can calculate this from Ix, and Iy 

"""
def get_keypoints(corners, Ix, Iy, threshold, diameter=50):
    
    def taninv(p,q):
        if p != 0:
            return np.degrees(np.arctan(q/p))
        else :
            return 90.0
    
    keypoints = []
    for i in range(corners.shape[0]):
        for j in range(corners.shape[1]):
            if corners[i,j] > threshold:
                theta = taninv(Iy[i,j],Ix[i,j])
                if Iy[i,j]>0 and Ix[i,j]>0:
                    theta = 360 - theta
                elif Iy[i,j]<0 and Ix[i,j]>0:
                    theta = (-1)*(theta)
                else:
                    theta = 180 - theta
             
                k = cv2.KeyPoint(j,i,diameter,theta,corners[i,j])
                keypoints.append(k)
       
    return keypoints


def get_features(image, kp=[], x=[], y=[], feature_width=16, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   kp: A list of coordinate objects for k number of keypoints
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """

    if kp==[]:
        assert len(x) == len(y)
        lx = x
        ly = y 
        
    if kp!=[]:
        ly,lx = zip(*[k.pt for k in kp])
        lx,ly = list(lx),list(ly)
        
#     #############################################################################
#     # TODO: YOUR CODE HERE                                                      #
#     # If you choose to implement rotation invariance, enabling it should not    #
#     # decrease your matching accuracy.                                          #
    
    def taninv(p,q):
        if p != 0:
            return np.degrees(np.arctan(q/p))
        else :
            return 90.0
    taninv1 = np.vectorize(taninv)
    
    d = int(feature_width/2)
    
    #print(d)
    fvlength = len(lx)
    Ix = cv2.Sobel(image,cv2.CV_64F,dx=1,dy=0)
    Iy = cv2.Sobel(image,cv2.CV_64F,dx=0,dy=1)
    mag = np.sqrt(Ix*Ix + Iy*Iy)
    ori = taninv1(Iy,Ix)%360
    image=np.pad(image,((d,d),(d,d)))
    mag=np.pad(mag,((d,d),(d,d)))
    ori=np.pad(ori,((d,d),(d,d)))
    
    ker = cv2.getGaussianKernel(feature_width,d)
    ker = ker @ ker.T
    
    fv = np.zeros((fvlength,128))
    i=0
    
    l = zip(lx,ly)
    for x,y in l:
        x,y = int(x),int(y)
        window = image[x:x+2*d,y:y+2*d]
        print("window = ",window.shape)
        
        mag_single = mag[x:x+2*d,y:y+2*d]
        mag_single = mag_single * ker
        ori_single = ori[x:x+2*d,y:y+2*d]
        print("ori_single = ", ori_single.shape)
            
        #mag = cv2.GaussianBlur(mag,(feature_width+1,feature_width+1),d)
        
        # orientation
        hist = []
        for k in range(36):
            hist.append(np.sum(mag_single[(10*k < ori_single) & (10*(k+1) > ori_single)]))
            
        pos = hist.index(np.max(hist))
        #print(hist)
        orientation = (pos*10+(pos+1)*10)/2
        
        # feature vector
        dd = int(feature_width/4)
        fv_single = []
        ori_single = ori_single - orientation
        ori_single[ori_single<0] = 360 + ori_single[ori_single<0]
        for k1 in range(4):
            for k2 in range(4):
                #fvwindow = image[dd*k2 : dd*(k2+1)-1, dd*k1 : dd*(k1+1)-1]
                mag_single1 = mag_single[dd*k2 : dd*(k2+1)-1, dd*k1 : dd*(k1+1)-1]
                print("subsets = ",mag_single1.shape)
                ori_single1 = ori_single[dd*k2 : dd*(k2+1)-1, dd*k1 : dd*(k1+1)-1]
                ldummy = [None]*8
                for k in range(8):
                    ldummy[k] = np.sum(mag_single1[(45*k < ori_single1) & (45*(k+1) > ori_single1)])
                fv_single.extend(ldummy)
            
        fv[i] = fv_single/np.sqrt(np.sum([item**2 for item in fv_single]))
        i=i+1
        
    return fv

        
                  
    ############################################################################

    ############################################################################
     #                           END OF YOUR CODE                              #
    ############################################################################
    