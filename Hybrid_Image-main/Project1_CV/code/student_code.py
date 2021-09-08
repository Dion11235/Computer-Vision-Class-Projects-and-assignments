import numpy as np
#### DO NOT IMPORT cv2 

def my_getGaussianKernel(ksize,sigma,mean=(0,0)):
    """
    Returns a 2D gaussian kernel (square/rectangular) with a custom mean, sigma.
    
    Args 
    - n : a tuple (a,b) which shall be the dimension of the produced kernel. 
          for a square kernel it's enough to give a single digit. 
    - sigma : standard deviation of the distribution.
    - mean : mean of the distribution. By default, set to (0,0).
    """
    n = ksize
    if isinstance(n,int):
        n = (n,n)
    m = np.zeros((n[0],n[1]))
    d = (n[0]//2, n[1]//2)
    
    for i in range(n[0]):
        for j in range(n[1]):
            x,y = i-d[0],j-d[1] 
            m[i][j] = np.exp((-(x-mean[0])**2-(y-mean[1])**2)/sigma**2)/(np.sqrt(2*np.pi)*(sigma))
    m = (1/(np.sum(m)))*(m)
    return m

def my_imfilter(image, filter):
    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    
    """
    Apply a filter to an image. Return the filtered image.

    Args
    - image: numpy nd-array of dim (m, n, c)
    - filter: numpy nd-array of dim (k, k)
              Returns
    - filtered_image: numpy nd-array of dim (m, n, c)

    HINTS:
    - You may not use any libraries that do the work for you. Using numpy to work
      with matrices is fine and encouraged. Using opencv or similar to do the
      filtering for you is not allowed.
    - I encourage you to try implementing this naively first, just be aware that
      it may take an absurdly long time to run. You will need to get a function
      that takes a reasonable amount of time to run so that I can verify
      your code works.
    - Remember these are RGB images, accounting for the final image dimension.
    """

  ############################
  ### TODO: YOUR CODE HERE ###
    
    if len(image.shape) == 2:
        image = np.stack((image,image,image),axis = 2)
    filtered_image=np.zeros(image.shape)
    for n in range(3):
        image1 = image[:,:,n]
        image_row, image_col = image1.shape
        kernel_row, kernel_col = filter.shape
        pad_width_col, pad_width_row = kernel_col // 2, kernel_row // 2 
        padded_image=np.pad(image1, ((pad_width_row,pad_width_row),(pad_width_col,pad_width_col)), mode='reflect')
        padded_image_row, padded_image_col=padded_image.shape
        for i in range(image_row):
            for j in range(image_col):
                filtered_image[i,j,n] = np.sum(filter*padded_image[i:i + kernel_row, j:j + kernel_col])
    
  ### END OF STUDENT CODE ####
  ############################

    return filtered_image

def create_hybrid_image(image1, image2, filter):
    
    """
    Takes two images and creates a hybrid image. Returns the low
    frequency content of image1, the high frequency content of
    image 2, and the hybrid image.

    Args
    - image1: numpy nd-array of dim (m, n, c)
    - image2: numpy nd-array of dim (m, n, c)
    Returns
    - low_frequencies: numpy nd-array of dim (m, n, c)
    - high_frequencies: numpy nd-array of dim (m, n, c)
    - hybrid_image: numpy nd-array of dim (m, n, c)

    HINTS:
    - You will use your my_imfilter function in this function.
    - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
    - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
    - If you want to use images with different dimensions, you should resize them
    in the notebook code.
    """
    if len(image1.shape) == 2:
        image1 = np.stack((image1,image1,image1),axis = 2)
    if len(image2.shape) == 2:
        image2 = np.stack((image2,image2,image2),axis = 2)
    
    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

  ############################
  ### TODO: YOUR CODE HERE ###
    
    s1=image1.shape
    s2=image2.shape
    low_frequencies = my_imfilter(image1, filter)
    high_frequencies = image2-my_imfilter(image2, filter)
    hybrid_image = np.add(low_frequencies,high_frequencies)
    hybrid_image = hybrid_image - np.min(hybrid_image)
    hybrid_image /= np.max(hybrid_image)
    
  ### END OF STUDENT CODE ####
  ############################

    return low_frequencies, high_frequencies, hybrid_image
