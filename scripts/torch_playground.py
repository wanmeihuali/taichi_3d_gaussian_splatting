import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F


# # What pytorch has to guess
# T_world_camera_true = torch.tensor([[1., 0., 0., 1.],
#                                     [0., 0., 1., 0.],
#                                     [0., -1., 0., 0.],
#                                     [0., 0., 0., 1.]])

# What pytorch has to guess
T_world_camera_true = torch.tensor([[1.0000000,  0.0000000,  0.0000000, 0.5],
                                    [0.0000000,  0.9961947, -0.0871558, 0.],
                                    [0.0000000,  0.0871558,  0.9961947 , 0.],
                                    [0., 0., 0., 1.]])


def quaternion_to_rotation_matrix_torch(q):
    """
    Convert a quaternion into a full three-dimensional rotation matrix.

    Input:
    :param q: A tensor of size (B, 4), where B is batch size and quaternion is in format (x, y, z, w).

    Output:
    :return: A tensor of size (B, 3, 3), where B is batch size.
    """
    # Ensure quaternion has four components
    assert q.shape[-1] == 4, "Input quaternion should have 4 components!"

    x, y, z, w = q.unbind(-1)

    # Compute quaternion norms
    q_norm = torch.norm(q, dim=-1, keepdim=True)
    # Normalize input quaternions
    q = q / q_norm

    # Compute rotation matrix
    rot_matrix = torch.empty(
        (*q.shape[:-1], 3, 3), dtype=q.dtype, device=q.device)
    rot_matrix[..., 0, 0] = 1 - 2 * (y**2 + z**2)
    rot_matrix[..., 0, 1] = 2 * (x*y - z*w)
    rot_matrix[..., 0, 2] = 2 * (x*z + y*w)
    rot_matrix[..., 1, 0] = 2 * (x*y + z*w)
    rot_matrix[..., 1, 1] = 1 - 2 * (x**2 + z**2)
    rot_matrix[..., 1, 2] = 2 * (y*z - x*w)
    rot_matrix[..., 2, 0] = 2 * (x*z - y*w)
    rot_matrix[..., 2, 1] = 2 * (y*z + x*w)
    rot_matrix[..., 2, 2] = 1 - 2 * (x**2 + y**2)

    return rot_matrix

def quaternion_to_rotation_matrix_torch_jacobian(q):
    qx, qy, qz, qw = q
    dR_dqx = torch.tensor([
        [0, 2*qy, 2*qz],
        [2*qy, -4*qx, -2*qw],
        [2*qz, 2*qw, -4*qx]
    ]) 
    dR_dqy = torch.tensor([
        [-4*qy, 2*qx, 2*qw],
        [2*qx, 0, 2*qz],
        [-2*qw, 2*qz, -4*qy]
    ])
    dR_dqz = torch.tensor([
        [-4*qz, -2*qw, 2*qx],
        [2*qw, -4*qz, 2*qy],
        [2*qx, 2*qy, 0]
    ])
    dR_dqw = torch.tensor([
        [0, -2*qz, 2*qy],
        [2*qz, 0, -2*qx],
        [-2*qy, 2*qx, 0]
    ])
    return  dR_dqx, dR_dqy, dR_dqz, dR_dqw

def inverse_SE3(transform: torch.Tensor):
    R = transform[:3, :3]
    t = transform[:3, 3]
    inverse_transform = torch.zeros_like(transform)
    inverse_transform[:3, :3] = R.T
    inverse_transform[:3, 3] = -R.T @ t
    inverse_transform[3, 3] = 1
    return inverse_transform

def render(
        point_cloud, 
        T_world_camera,
        projective_transform,
        image_size: tuple
    ):        


    T_camera_world = inverse_SE3(T_world_camera)
    if T_camera_world.dtype != point_cloud.dtype:
            # Convert T_camera_world to the data type of lidar_point_cloud_homogeneous
            T_camera_world = T_camera_world.to(point_cloud.dtype)
            
    transformed_points = torch.matmul(T_camera_world.cuda(), point_cloud)
    
    transformed_points = torch.reshape(transformed_points, (4, -1))
    
    transformed_points = transformed_points[:3, :]
    if projective_transform.dtype != transformed_points.dtype:
        # Convert T_camera_world to the data type of lidar_point_cloud_homogeneous
        projective_transform = projective_transform.to(transformed_points.dtype)        
    uv1 = torch.matmul(projective_transform, transformed_points) / \
        transformed_points[ 2, :]
    uv = uv1[:2, :]
    return uv

    u = torch.floor(uv[0, :]).long()
    v = torch.floor(uv[1, :]).long()


    depth_map = torch.full((image_size[1], image_size[0]), -1.0, dtype=point_cloud.dtype,device="cuda")
    depth_map[v[:], u[:]] = point_cloud[:, 2]

    return depth_map


def jacobian_t_R(R):
    # Dimensions of the rotation matrix
    n = R.shape[0]

    # Initialize the derivative matrix
    dT_dR = np.zeros((n + 1, n, n))

    # Compute the derivative for each element of R
    for i in range(n):
        for j in range(n):
            dR_ij = np.zeros((n, n))
            dR_ij[i, j] = 1  # Perturb the (i, j)-th element
            dT_dR[:, i, j] = np.reshape(dR_ij, (n * n,))
    
    return dT_dR

def project_to_camera_position_jacobian(
        w_t_w_points,
        T_camera_world,
        projective_transform,
    ):
        T = T_camera_world
        W = torch.tensor([
            [T[0, 0], T[0, 1], T[0, 2]],
            [T[1, 0], T[1, 1], T[1, 2]],
            [T[2, 0], T[2, 1], T[2, 2]]
        ])

        # t = T_camera_world @ torch.tensor(
        #    [w_t_w_points[0], w_t_w_points[1], w_t_w_points[2], 1])
        t = torch.matmul(T_camera_world, w_t_w_points)
        K = projective_transform

        # d_uv_d_translation_camera = torch.empty(
        #             size=(w_t_w_points.shape[0], 2, 3), dtype=torch.float32, device="cuda")

        d_uv_d_translation_camera = torch.tensor([
            [K[0, 0] / t[2], K[0, 1] / t[2], (-K[0, 0] * t[0] - K[0, 1] * t[1]) / (t[2] * t[2])],
            [K[1, 0] / t[2], K[1, 1] / t[2], (-K[1, 0] * t[0] - K[1, 1] * t[1]) / (t[2] * t[2])]])
        
        return d_uv_d_translation_camera
    
        d_translation_camera_d_translation = W
        d_uv_d_translation = d_uv_d_translation_camera @ d_translation_camera_d_translation  # 2 x 3
        return d_uv_d_translation
    
class TransformModel(torch.autograd.Function):

    @staticmethod
    def forward(ctx, w_t_w_point, q_w_c_guess, t_w_c_guess):
        q_w_c_guess = F.normalize(q_w_c_guess, p=2, dim=-1)
        R = quaternion_to_rotation_matrix_torch(q_w_c_guess).cuda()
        T_w_c_guess = torch.vstack((torch.hstack((R, t_w_c_guess)),
                                    torch.tensor([0., 0., 0., 1.], device="cuda")))
        ctx.save_for_backward(q_w_c_guess, t_w_c_guess, w_t_w_point)
        uv_pred = render(w_t_w_point, T_w_c_guess, K, (720, 405))
        return uv_pred

    @staticmethod
    def backward(ctx, grad_uv_pred):
        q_w_c_guess, t_w_c_guess, w_t_w_point= ctx.saved_tensors
        R_guess = quaternion_to_rotation_matrix_torch(q_world_camera_initial_guess)
        T_w_c_guess = torch.vstack((torch.hstack((R_guess, t_world_camera_initial_guess)),
                                torch.tensor([0., 0., 0., 1.], device="cuda")))
        T_c_w_guess = inverse_SE3(T_w_c_guess)
        
        grad_q = torch.zeros((4), device="cuda")  # Your gradient computation for q
        grad_t = torch.zeros((3, 1), device="cuda")   # Your gradient computation for t
        d_uv_d_translation_camera = project_to_camera_position_jacobian(w_t_w_point, T_c_w_guess, K).cuda()
        
        dR_dqx, dR_dqy, dR_dqz, dR_dqw = quaternion_to_rotation_matrix_torch_jacobian((q_w_c_guess[0], q_w_c_guess[1],q_w_c_guess[2],q_w_c_guess[3]))
        
        # x, y, z: coordinate of points in camera frame
        dx_dq = torch.tensor([[dR_dqx[0,0]*w_t_w_point[0] + dR_dqx[1,0]*w_t_w_point[1] + dR_dqx[2,0]*w_t_w_point[2]],
                            [dR_dqy[0,0]*w_t_w_point[0] + dR_dqy[1,0]*w_t_w_point[1] + dR_dqy[2,0]*w_t_w_point[2]],
                            [dR_dqz[0,0]*w_t_w_point[0] + dR_dqz[1,0]*w_t_w_point[1] + dR_dqz[2,0]*w_t_w_point[2]],
                            [dR_dqw[0,0]*w_t_w_point[0] + dR_dqw[1,0]*w_t_w_point[1] + dR_dqw[2,0]*w_t_w_point[2]]], 
                            device="cuda")
                    
        dy_dq = torch.tensor([[dR_dqx[0,1]*w_t_w_point[0] + dR_dqx[1,1]*w_t_w_point[1] + dR_dqx[2,1]*w_t_w_point[2]],
                            [dR_dqy[0,1]*w_t_w_point[0] + dR_dqy[1,1]*w_t_w_point[1] + dR_dqy[2,1]*w_t_w_point[2]],
                            [dR_dqz[0,1]*w_t_w_point[0] + dR_dqz[1,1]*w_t_w_point[1] + dR_dqz[2,1]*w_t_w_point[2]],
                            [dR_dqw[0,1]*w_t_w_point[0] + dR_dqw[1,1]*w_t_w_point[1] + dR_dqw[2,1]*w_t_w_point[2]]],
                             device="cuda")
        
        dz_dq = torch.tensor([[dR_dqx[0,2]*w_t_w_point[0] + dR_dqx[1,2]*w_t_w_point[1] + dR_dqx[2,2]*w_t_w_point[2]],
                            [dR_dqy[0,2]*w_t_w_point[0] + dR_dqy[1,2]*w_t_w_point[1] + dR_dqy[2,2]*w_t_w_point[2]],
                            [dR_dqz[0,2]*w_t_w_point[0] + dR_dqz[1,2]*w_t_w_point[1] + dR_dqz[2,2]*w_t_w_point[2]],
                            [dR_dqw[0,2]*w_t_w_point[0] + dR_dqw[1,2]*w_t_w_point[1] + dR_dqw[2,2]*w_t_w_point[2]]],
                             device="cuda")
        
        d_translation_camera_d_q = torch.vstack((torch.transpose(dx_dq, 0, 1),
                                                torch.transpose(dy_dq, 0, 1),
                                                torch.transpose(dz_dq, 0, 1)))
        
        dxyz_d_t_world_camera = -R_guess.T
        grad_q = d_uv_d_translation_camera @ d_translation_camera_d_q
        
        grad_t = d_uv_d_translation_camera @ dxyz_d_t_world_camera
        # print("gradient_uv_t_world_camera")
        # print(grad_t)
        return None, grad_uv_pred.T @ grad_q, (grad_uv_pred.T @ grad_t).T
            
    
K = torch.tensor([[400, 0, 360],
                  [0, 400, 202.5],
                  [0, 0, 1]], device="cuda")


image_gt = torch.full((405, 720), 0.,device="cuda")
image_gt[210, 50] = 1
image_gt[50, 350] = 2
image_gt[380, 700] = 3

# Given data
uv_a = torch.tensor([50, 210])
uv_b = torch.tensor([350, 50])
uv_c = torch.tensor([700, 380])

depth_a = 1
depth_b = 2
depth_c = 3

uv_groundtruth = torch.tensor([[50, 210],
                               [350, 50],
                               [700, 380]], device="cuda")

c_t_c_a = torch.tensor([[(uv_a[0]-K[0,2])*depth_a/K[0,0]],
                        [(uv_a[1]-K[1,2])*depth_a/K[1,1]],
                        [depth_a]])


c_t_c_b = torch.tensor([[(uv_b[0]-K[0,2])*depth_b/K[0,0]],
                        [(uv_b[1]-K[1,2])*depth_b/K[1,1]],
                        [depth_b]])

c_t_c_c = torch.tensor([[(uv_c[0]-K[0,2])*depth_c/K[0,0]],
                        [(uv_c[1]-K[1,2])*depth_c/K[1,1]],
                        [depth_c]])

w_t_w_a = torch.matmul(T_world_camera_true, torch.vstack((c_t_c_a,  torch.tensor(1))).to(T_world_camera_true.dtype))
w_t_w_b = torch.matmul(T_world_camera_true, torch.vstack((c_t_c_b,  torch.tensor(1))).to(T_world_camera_true.dtype))
w_t_w_c = torch.matmul(T_world_camera_true, torch.vstack((c_t_c_c,  torch.tensor(1))).to(T_world_camera_true.dtype))

w_t_w_points = torch.hstack((w_t_w_a, w_t_w_b, w_t_w_c)).cuda()

image_gt = image_gt / torch.max(image_gt)

q_world_camera_initial_guess = torch.tensor([0.,0.,0., 1.], device="cuda", requires_grad=True)
t_world_camera_initial_guess = torch.tensor([[0.],
                                             [0.],
                                             [0.]], device="cuda", requires_grad=True)

# Instantiate the model, loss function, and optimizer
model = TransformModel()
rotation_optimizer = torch.optim.Adam(
            [q_world_camera_initial_guess], lr=0.00005)
optimizer = torch.optim.Adam(
            [q_world_camera_initial_guess, t_world_camera_initial_guess], lr=0.0001)

# Training loop
num_epochs = 100_000
for epoch in range(num_epochs):
    # One point at the time?
    for i in range(3):
        w_t_w_point = w_t_w_points[:,i]
        uv_groundtruth_point = uv_groundtruth[i, :]
        optimizer.zero_grad()
        
        # Forward pass
        model = TransformModel.apply
        render_pred = model(w_t_w_point, q_world_camera_initial_guess, t_world_camera_initial_guess) # IT'S WRONGGGGGG
        render_pred = torch.reshape(render_pred, (-1,2))
        
        # Compute the loss
        loss = (render_pred - uv_groundtruth_point.to(render_pred.dtype)).pow(2).sum()

        # loss = torch.abs(render_pred - image_gt).mean()

        # Backward pass and optimization
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(q_world_camera_initial_guess, 50000)
        torch.nn.utils.clip_grad_norm_(t_world_camera_initial_guess, 50000)
        # rotation_optimizer.step()
        # translation_optimizer.step()
        optimizer.step()

        # with torch.no_grad():
        #     q_world_camera_initial_guess = F.normalize(q_world_camera_initial_guess, p=2, dim=-1)
    # Print the loss every 100 epochs
    if (epoch + 1) % 1000 == 0:
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        print("Current guess")
        print("q_world_camera_initial_guess:")
        print(q_world_camera_initial_guess)
        print("t_world_camera_initial_guess:")
        print(t_world_camera_initial_guess)
        R = quaternion_to_rotation_matrix_torch(q_world_camera_initial_guess)
        T_w_c_guess = torch.vstack((torch.hstack((R, t_world_camera_initial_guess)),
                                    torch.tensor([0., 0., 0., 1.], device="cuda")))
        print("T_w_c_guess:")
        print(T_w_c_guess)
        
with torch.no_grad():
    render_pred = model(w_t_w_point, q_world_camera_initial_guess, t_world_camera_initial_guess).cuda()
    render_pred = torch.transpose(render_pred, 0, 1)