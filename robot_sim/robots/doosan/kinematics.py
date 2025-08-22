'''
ikpy

- reference: https://ikpy.readthedocs.io/en/latest/chain.html#ikpy.chain.Chain.inverse_kinematics_frame
'''

from ikpy.chain import Chain
import numpy as np 

class Kinematics:
    def __init__(self, urdf_file, 
                    base_elements=['base_link'],
                    active_links_mask=[False, True, True, True, True, True, True, False]):
        self.chain = Chain.from_urdf_file(urdf_file,
                                    base_elements=base_elements,
                                    # base_element_type="link",
                                    active_links_mask=active_links_mask,
                                )


    @property
    def links(self):
        return [l.name for l in self.chain.links]
        
    def fk(self, joints, full_kinematics=False, return_only_position=True):
        
        if len(joints) == 6:
            joints = np.concatenate(([0], joints, [0]))
        
        pos = self.chain.forward_kinematics(joints, full_kinematics=full_kinematics)
        
        if return_only_position:
            return pos[:3, 3]

        return pos
    
    def ik(self, target_position, **kwargs):
        return self.chain.inverse_kinematics(target_position=target_position, **kwargs)

    def ik_frame(self, target, **kwargs):
        
        joints = self.chain.inverse_kinematics_frame(target=target, **kwargs)
        return joints[1:-1]
    
    
if __name__ == '__main__':
    import numpy as np
    
    k = Kinematics('/HDD/_projects/github/robot/data/descriptions/doosan/m1013_modified_v2.urdf',
                   base_elements=['base_link'])
    print('links: ', k.links)
    
    target_position = [0.46401721239089966, 0.0345260351896286, 0.7413025498390198]
    kwargs = {'optimizer': 'least_squares'}
    print("1-1. The angles of each joints are : ", k.ik(target_position=target_position,
                                                                    target_orientation=None, 
                                                                    orientation_mode=None,
                                                                    **kwargs))
    rotation = np.eye(3)
    transformation_matrix = np.vstack([np.hstack([rotation, np.array(target_position).reshape(-1, 1)]), [0, 0, 0, 1]])

    print("1-2. The angles of each joints are : ", k.ik_frame(target=transformation_matrix, orientation_mode="all"))
    
    real_frame = k.fk(k.ik(target_position))
    print("Computed position vector : %s, original position vector : %s" % (real_frame, target_position))
    
    
    import pandas as pd 
    import ast
    from scipy.spatial.transform import Rotation as R

    traj = pd.read_csv('/HDD/etc/outputs/isaac/trajectory.csv')
    joint_position = traj['joint_position'].apply(ast.literal_eval)
    joint_velocity = traj['joint_velocity'].apply(ast.literal_eval)
    ee_pos_w = traj['ee_pos_w'].apply(ast.literal_eval)
    ee_quat_w_xyzw = traj['ee_quat_w_xyzw'].apply(ast.literal_eval)

    idx = 1

    print('first joint: ', joint_position[idx])
    print('first ee-pos: ', ee_pos_w[idx])
    print('first ee-quat: ', ee_quat_w_xyzw[idx])

    ik_position = k.ik(target_position=ee_pos_w[idx])
    print('a. ik_position: ', ik_position)
    fk_position = k.fk(ik_position)
    print('a. fk_position: ', fk_position)
    
    # rotation = R.from_quat([ee_quat_w_xyzw[idx][3], ee_quat_w_xyzw[idx][0], 
    #                         ee_quat_w_xyzw[idx][1], ee_quat_w_xyzw[idx][2]
    #                     ]).as_matrix() 
    # rotation = R.from_quat(ee_quat_w_xyzw[idx]).as_matrix() # [ 0.00000000e+00  9.95565006e-07 -1.74444374e-01  1.57000056e+00
                                                              #   1.32965328e-06  1.56999880e+00  0.00000000e+00  0.00000000e+00]
    rotation = np.eye(3)
    transformation_matrix = np.vstack([np.hstack([rotation, np.array(ee_pos_w[0]).reshape(-1, 1)]), [0, 0, 0, 1]])
    ik_position_frame = k.ik_frame(target=transformation_matrix, 
                                   initial_position=[0] + joint_position[idx -1] + [0])
    print('b. ik_position_frame: ', ik_position_frame)
    fk_position_frame = k.fk(ik_position_frame)
    print('b. fk_position_frame: ', fk_position_frame)


    
    