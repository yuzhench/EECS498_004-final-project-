import numpy as np

def angle_between_vectors(v1, v2):
    # Normalize (just to be safe)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    
    # Clip to avoid numerical issues with arccos
    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# Your vectors
gt_normal = np.array([ -0.791647, 0.15893749,  -0.61934179 ])
# final_normal = np.array([ 0.08930327,  0.31278366, -0.94561689 ]) #1163
# final_normal = np.array([-0.17330183,  0.31645299, -0.93264354]) #080
final_normal = np.array([-0.77862889,  0.18620399, -0.59922043]) #380

# Calculate mismatch angle
mismatch_angle = angle_between_vectors(gt_normal, final_normal)
print(f"Angle between GT normal and final transformed GT normal: {mismatch_angle:.2f} degrees")
