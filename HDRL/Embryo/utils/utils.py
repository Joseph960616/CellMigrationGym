import numpy as np
# from scipy.spatial.distance import pdist, squareform

def rgb2gray(rgb):
    """
    Converting RGB image into Gray image
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def gray2twobit(gray):
    """
    Converting gray image into 2bit data map
    """
    twobit = np.zeros((gray.shape[0],gray.shape[1]))
    for row in range(len(gray.shape[0])):
        for col in range(len(gray.shape[1])):
            if gray[row][col] < 64:
                twobit[row][col] = 0
            elif gray[row][col] >= 64 and gray[row][col] < 128:
                twobit[row][col] = 1
            elif gray[row][col] >= 128 and gray[row][col] < 192:
                twobit[row][col] = 2
            else:
                twobit[row][col] = 3
    return twobit

def depth_conversion(depth_buffer):
    """
    Get the real depth from the depth buffer with comes from pybullet getCameraImage API
    """
    row,col = depth_buffer.shape
    depth = np.zeros((row,col))
    far = 1000.
    near = 0.01
    for i in range(row):
        for j in range(col):
            depth[i,j] = far * near / (far - (far - near) * depth_buffer[i,j])
    return depth

def embryo_volume(data_a):
    # x = data_s[:,0]
    # y = data_s[:,1]
    # z = data_s[:,2]
    volume_a = []
    for data_s in data_a:
        axis_lenth = []
        for i in range(3):
            data = data_s[:,i]
            index = [np.where(data == min(data))[0][0],np.where(data == max(data))[0][0]]
            # print(index)
            axis_lenth.append(np.linalg.norm(data[index[0]]-data[index[1]]))
        volume = 4.0/3 * np.pi * \
                axis_lenth[0] * \
                axis_lenth[1] * \
                axis_lenth[2]
        volume_a.append(volume)
        # print(volume)
    
    return max(volume_a)
        
# def maxdistance(data):
#     D = pdist(data)
#     D = squareform(D);
#     N, [I_row, I_col] = np.nanmax(D), np.unravel_index( np.argmax(D), D.shape )
#     return N, [I_row, I_col]
            


