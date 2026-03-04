#####################################################################################
## Rotation related manipulations
#####################################################################################

#%% 
# Notation based on Introduction to Continuum Mechanics by Lai, Rubin, Krempl, 4th ed., 2010, Elsevier.
# equation 2.16.5 in the book
def define_rotation_tensor_from_vectors(v1,v2,v3):
    import numpy as np
    return np.stack((v1,v2,v3),axis=4)

#%% rotation of 1D arrray
# Notation based on Introduction to Continuum Mechanics by Lai, Rubin, Krempl, 4th ed., 2010, Elsevier.
# equation 2.17.6 in the book
def rotate_1D_tensor(T1,Qtrans,Q):
    import numpy as np
    return np.matmul(Qtrans,T1)

#%%
# Notation based on Introduction to Continuum Mechanics by Lai, Rubin, Krempl, 4th ed., 2010, Elsevier.
# equation 2.18.5 in the book
def rotate_2D_tensor(T2,Qtrans,Q):
    import numpy as np
    return np.matmul( np.matmul(Qtrans,T2) , Q)

#%%
def convert_2Dtensor_from_6c_to_3x3(T2_6c):
    import numpy as np

    T2_9c = T2_6c[ ..., [0,3,4 , 3,1,5 , 4,5,2] ]
    T2_9c = T2_9c.reshape( T2_9c.shape[0:-1]+(3,3) , order="F" )
    return T2_9c

#%%
def convert_1Dtensor_from_3c_to_3x1(T1):
    import numpy as np

    T1_3x1 = T1[..., np.newaxis]
    return T1_3x1

#%%
def convert_2Dtensor_from_3x3_to_6c(T2_3x3):
    import numpy as np

    T2_6c = T2_3x3.reshape( T2_3x3.shape[:-2]+(9,) , order="F" )
    T2_6c = T2_6c[...,[0,4,8,1,2,5]]
    return T2_6c

#%% helper functions defined for convenience
def rotate_6c2D_tensor(T2,Qtrans,Q):
    T2_rot = convert_2Dtensor_from_6c_to_3x3(T2)
    T2_rot = rotate_2D_tensor(T2_rot,Qtrans,Q)
    T2_rot = convert_2Dtensor_from_3x3_to_6c(T2_rot)
    return T2_rot

def rotate_vector(V,Qtrans,Q):
    V_rot = convert_1Dtensor_from_3c_to_3x1(V)
    V_rot = rotate_1D_tensor(V_rot,Qtrans,Q)
    V_rot = V_rot[...,0]
    return V_rot

