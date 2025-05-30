import torch

################################################################################
## READ/WRITE TO FILES
################################################################################

import json

def read_json(absolute_filepath):
	with open(absolute_filepath, 'r') as f:
		data = json.load(f)
	return data

def write_json(data, absolute_filepath, pretty=False):
	with open(absolute_filepath, "w") as f:
		if pretty:
			json.dump(data, f, ensure_ascii=False, indent=2)
		else:
			json.dump(data, f)


################################################################################
## ANGLE TRANSFORMATION FONCTIONS
################################################################################

import roma

def rotvec_to_eulerangles(x):
	x_rotmat = roma.rotvec_to_rotmat(x)
	thetax = torch.atan2(x_rotmat[:,2,1], x_rotmat[:,2,2])
	thetay = torch.atan2(-x_rotmat[:,2,0], torch.sqrt(x_rotmat[:,2,1]**2+x_rotmat[:,2,2]**2))
	thetaz = torch.atan2(x_rotmat[:,1,0], x_rotmat[:,0,0])
	return thetax, thetay, thetaz

def eulerangles_to_rotmat(thetax, thetay, thetaz):
	N = thetax.numel()
	# rotx
	rotx = torch.eye( (3) ).to(thetax.device).repeat(N,1,1)
	roty = torch.eye( (3) ).to(thetax.device).repeat(N,1,1)
	rotz = torch.eye( (3) ).to(thetax.device).repeat(N,1,1)
	rotx[:,1,1] = torch.cos(thetax)
	rotx[:,2,2] = torch.cos(thetax)
	rotx[:,1,2] = -torch.sin(thetax)
	rotx[:,2,1] = torch.sin(thetax)
	roty[:,0,0] = torch.cos(thetay)
	roty[:,2,2] = torch.cos(thetay)
	roty[:,0,2] = torch.sin(thetay)
	roty[:,2,0] = -torch.sin(thetay)
	rotz[:,0,0] = torch.cos(thetaz)
	rotz[:,1,1] = torch.cos(thetaz)
	rotz[:,0,1] = -torch.sin(thetaz)
	rotz[:,1,0] = torch.sin(thetaz)
	rotmat = torch.einsum('bij,bjk->bik', rotz, torch.einsum('bij,bjk->bik', roty, rotx))
	return rotmat

def eulerangles_to_rotvec(thetax, thetay, thetaz):
	rotmat = eulerangles_to_rotmat(thetax, thetay, thetaz)
	return roma.rotmat_to_rotvec(rotmat)


################################################################################
## LOAD POSE DATA
################################################################################

import os
import numpy as np

import text2pose.config as config

def process_relative_pose(pose):
    # 选择第一个点作为参考点
    reference_point = pose[0]
    
    # 计算所有点相对于参考点的偏移量
    relative_pose = pose - reference_point
    
    return relative_pose

def get_pose_data_from_file(pose_info, applied_rotation=None, output_rotation=False):
	"""
	Load pose data and normalize the orientation.

	Args:
		pose_info: list [dataset (string), sequence_filepath (string), frame_index (int)]
		applied_rotation: rotation to be applied to the pose data. If None, the
			normalization rotation is applied.
		output_rotation: whether to output the rotation performed for
			normalization, in addition of the normalized pose data.

	Returns:
		pose data, torch.tensor of size (1, n_joints*3), all joints considered.
		(optional) R, torch.tensor representing the rotation of normalization
	"""

	# load pose data
	assert pose_info[0] in config.supported_datasets, f"Expected data from on of the following datasets: {','.join(config.supported_datasets)} (provided dataset: {pose_info[0]})."
	
	# if pose_info[0] == "AMASS":
	# 	dp = np.load(os.path.join(config.supported_datasets[pose_info[0]], pose_info[1]))
	# 	pose = dp['poses'][pose_info[2],:].reshape(-1,3) # (n_joints, 3)
	# 	pose = torch.as_tensor(pose).to(dtype=torch.float32)
 
	# 20250325 dzj
	# import pdb
	# pdb.set_trace()
	if pose_info[0] == "AMASS":
		dp = np.load(os.path.join(config.supported_datasets[pose_info[0]], pose_info[1]))
		# pose = dp['poses'][pose_info[2],:].reshape(-1,3) # (n_joints, 3)
		pose = dp[pose_info[2],:(config.NB_INPUT_JOINTS),:]
		pose = process_relative_pose(pose) ## all relative calculate
		pose = torch.as_tensor(pose).to(dtype=torch.float32)
		# pose = pose[:config.NB_INPUT_JOINTS]  # 20250325 dzj [22, 3]
	elif pose_info[0] == "MOTIONX":
		dp = np.load(os.path.join(config.supported_datasets[pose_info[0]], pose_info[1]))
		# pose = dp[pose_info[2],:(config.NB_INPUT_JOINTS * 3)].reshape(-1,3) # (n_joints, 3)
		pose = dp[pose_info[2],:(config.NB_INPUT_JOINTS),:]
		pose = process_relative_pose(pose) ## all relative calculate
		pose = torch.as_tensor(pose).to(dtype=torch.float32)
		# pose = pose[:config.NB_INPUT_JOINTS]  # 20250325 dzj[22, 3]
	# 20250325 dzj

    ####################################  dzj joints process###############################
	# # normalize the global orient
	# initial_rotation = pose[:1,:].clone()
	# if applied_rotation is None:
	# 	thetax, thetay, thetaz = rotvec_to_eulerangles( initial_rotation )
	# 	zeros = torch.zeros_like(thetaz)
	# 	pose[0:1,:] = eulerangles_to_rotvec(thetax, thetay, zeros)
	# else:
	# 	pose[0:1,:] = roma.rotvec_composition((applied_rotation, initial_rotation))
	# if output_rotation:
	# 	# a = A.u, after normalization, becomes a' = A'.u
	# 	# we look for the normalization rotation R such that: a' = R.a
	# 	# since a = A.u ==> u = A^-1.a
	# 	# a' = A'.u = A'.A^-1.a ==> R = A'.A^-1
	# 	R = roma.rotvec_composition((pose[0:1,:], roma.rotvec_inverse(initial_rotation)))
	# 	return pose.reshape(1, -1), R
	
	# return pose.reshape(1, -1)

	return pose.reshape(1,-1)


def pose_data_as_dict(pose_data, code_base='human_body_prior'):
	"""
	Args:
		pose_data, torch.tensor of shape (*, n_joints*3) or (*, n_joints, 3),
			all joints considered.
	Returns:
		dict
	"""
	# reshape to (*, n_joints*3) if necessary
	if len(pose_data.shape) == 3:
		# shape (batch_size, n_joints, 3)
		pose_data = pose_data.flatten(1,2)
	if len(pose_data.shape) == 2 and pose_data.shape[1] == 3:
		# shape (n_joints, 3)
		pose_data = pose_data.view(1, -1)
	# provide as a dict, with different keys, depending on the code base
	if code_base == 'human_body_prior':
		d = {"root_orient":pose_data[:,:3],
	   		 "pose_body":pose_data[:,3:66]}
		if pose_data.shape[1] > 66:
			d["pose_hand"] = pose_data[:,66:]
	elif code_base == 'smplx':
		d = {"global_orient":pose_data[:,:3],
	   		 "body_pose":pose_data[:,3:66]}
		if pose_data.shape[1] > 66:
			d.update({"left_hand_pose":pose_data[:,66:111],
					"right_hand_pose":pose_data[:,111:]})
	return d


if __name__ == "__main__":
	import numpy as np
	import torch
	# Visualize your final data, please define your example_path, like 'new_data_humanml_000067_joints_using_smplx_rotation.npy'
	example_path = '/scratch/project_465000903/LLava_video/ALL_except_motion_model/PoseClip/AMASS_smplh/KIT/3/jump_left02_poses.npz'
	assert example_path != None
	joints = np.load(example_path)
	# import pdb 
	# pdb.set_trace()

	############################smplh
	poses = joints['poses']
	joints = poses.reshape(-1,52,3)

	###########################smplx
	# joints = joints[:,:66]
	# joints = joints.reshape(-1,22,3)
	pose = joints[0]
	pose = torch.tensor(pose)
	initial_rotation = pose[:1,:].clone()
	# if applied_rotation is None:
	thetax, thetay, thetaz = rotvec_to_eulerangles( initial_rotation )
	zeros = torch.zeros_like(thetaz)
	pose[0:1,:] = eulerangles_to_rotvec(thetax, thetay, zeros)

	# import pdb
	# pdb.set_trace()

	# # 2*3*5=30, left first, then right
	# hand_joints_id = [i for i in range(25, 55)]
	# body_joints_id = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
	#                   12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # 22 joints

	# t2m_kinematic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [
	#     0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21], [9, 13, 16, 18, 20]]
	# t2m_left_hand_chain = [[20, 22, 23, 24], [20, 34, 35, 36], [
	#     20, 25, 26, 27], [20, 31, 32, 33], [20, 28, 29, 30]]
	# t2m_right_hand_chain = [[21, 43, 44, 45], [21, 46, 47, 48], [
	#     21, 40, 41, 42], [21, 37, 38, 39], [21, 49, 50, 51]]
	# t2m_body_hand_kinematic_chain = t2m_kinematic_chain + \
	#     t2m_left_hand_chain + t2m_right_hand_chain

	# if joints.shape[1] != 52:
	#     joints = joints[:, body_joints_id+hand_joints_id, :]
	# xyz = joints.reshape(1, -1, 52, 3)
	# xyz = xyz.squeeze(0)

	# # xyz = joints.reshape(-1,22,3)
	# xyz /= np.abs(xyz).max()


	xyz = joints
	xyz = xyz[:,:22,:]
	plot_3d_motion_with_ground(xyz)
	# pose_vis = draw_to_batch_smplh(xyz, t2m_body_hand_kinematic_chain, title_batch=None, outname=[
	#                                f'output.gif'])

