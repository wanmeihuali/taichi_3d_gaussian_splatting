# %%
from Camera import CameraInfo
from utils import get_ray_origin_and_direction_from_camera, get_ray_origin_and_direction_from_camera_by_gpt
from utils import torch_single_point_alpha_forward
import scipy.spatial.transform as transform
import taichi as ti
import torch
import numpy as np
import sympy
from sympy import latex, pprint
# %%
"""
J = projective_transform_jacobian
T = T_camera_world
R = rotation_matrix_from_quaternion(self.cov_rotation)
S = self.cov_scale
Sigma = R @ S.outer_product(S) @ R.transpose()

homogeneous_translation_camera = T_camera_world @ ti.math.vec4(self.translation.x, self.translation.y, self.translation.z, 1)
translation_camera = ti.math.vec3(homogeneous_translation_camera.x, homogeneous_translation_camera.y, homogeneous_translation_camera.z)
uv1 = (projective_transform @ translation_camera) / translation_camera.z
uv = ti.math.vec2(uv1.x, uv1.y)
"""
"""
T is 4x4
projective_transform is 3x3
get Jacobian of D(uv)/D(translation)
"""
# %%
projective_transform = sympy.MatrixSymbol('K', 3, 3)
translation_camera = sympy.MatrixSymbol('t', 3, 1)
uv1 = (projective_transform @ translation_camera) / translation_camera[2, 0]
uv = sympy.Matrix([uv1[0, 0], uv1[1, 0]])
D_uv_D_translation_camera = uv.jacobian(translation_camera)
D_uv_D_translation_camera.simplify()
print(latex(D_uv_D_translation_camera))
pprint(D_uv_D_translation_camera, use_unicode=True)
# %%
T = sympy.MatrixSymbol('T', 4, 4)
translation = sympy.MatrixSymbol('t', 3, 1)
homogeneous_translation_camera = T @ sympy.Matrix(
    [translation[0, 0], translation[1, 0], translation[2, 0], 1])
translation_camera = sympy.Matrix([homogeneous_translation_camera[0, 0],
                                  homogeneous_translation_camera[1, 0], homogeneous_translation_camera[2, 0]])

D_translation_camrea_D_translation = translation_camera.jacobian(translation)
D_translation_camrea_D_translation.simplify()
print(latex(D_translation_camrea_D_translation))
pprint(D_translation_camrea_D_translation, use_unicode=True)

# %%

uv1 = (projective_transform @ translation_camera) / translation_camera[2, 0]
uv = sympy.Matrix([uv1[0, 0], uv1[1, 0]])


D_uv_D_translation = uv.jacobian(translation)
D_uv_D_translation.simplify()
print(latex(D_uv_D_translation))
pprint(D_uv_D_translation, use_unicode=True)
# %%
"""
U = J @ W # 2x3
cov_uv = U @ Sigma @ U.transpose() # equation (5) in the paper
"""
U = sympy.MatrixSymbol('U', 2, 3)
Sigma = sympy.MatrixSymbol('Sigma', 3, 3)
Sigma_vec = sympy.Matrix([[Sigma[i, j]] for i in range(3) for j in range(3)])
cov_uv = U @ Sigma @ U.transpose()
cov_uv_vec = sympy.Matrix([[cov_uv[i, j]] for i in range(2) for j in range(2)])
D_cov_uv_D_U = cov_uv_vec.jacobian(Sigma_vec)
print(D_cov_uv_D_U.shape)
print(latex(D_cov_uv_D_U))
pprint(D_cov_uv_D_U, use_unicode=True)


# %%
"""
Sigma = M @ M.transpose() # covariance matrix, 3x3, equation (6) in the paper
"""
"""
M = sympy.MatrixSymbol('M', 3, 3)
Sigma = M @ M.transpose()
D_Sigma_D_M = Sigma.diff(M).as_explicit()
print(D_Sigma_D_M.shape)
print(latex(D_Sigma_D_M))
pprint(D_Sigma_D_M, use_unicode=True)
"""
M = sympy.Matrix([[sympy.Symbol(f"m{i}{j}")
                 for j in range(3)] for i in range(3)])
M_vector = sympy.Matrix([[M[i, j]] for i in range(3) for j in range(3)])

Sigma = M @ M.transpose()
Sigma_vector = sympy.Matrix([[Sigma[i, j]]
                            for i in range(3) for j in range(3)])
D_Sigma_D_M = Sigma_vector.jacobian(M_vector)
print(D_Sigma_D_M.shape)
print(latex(D_Sigma_D_M))
pprint(D_Sigma_D_M, use_unicode=True)
# %%
# M = R @ S
R = sympy.MatrixSymbol('R', 3, 3)
S = sympy.Matrix([
    [sympy.Symbol("s00"), 0, 0],
    [0, sympy.Symbol("s11"), 0],
    [0, 0, sympy.Symbol("s22")]
])
s_vec = sympy.Matrix([S[i, i] for i in range(3)])
M = R @ S
M_vector = sympy.Matrix([[M[i, j]] for i in range(3) for j in range(3)])
D_M_D_s = M_vector.jacobian(s_vec)
print(D_M_D_s.shape)
print(latex(D_M_D_s))
pprint(D_M_D_s, use_unicode=True)
# %%
"""

@ti.func
def rotation_matrix_from_quaternion(q: ti.math.vec4) -> ti.math.mat3:
    xx = q.x * q.x
    yy = q.y * q.y
    zz = q.z * q.z
    xy = q.x * q.y
    xz = q.x * q.z
    yz = q.y * q.z
    wx = q.w * q.x
    wy = q.w * q.y
    wz = q.w * q.z
    return ti.math.mat3([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ])
 
"""
x = sympy.Symbol('x')
y = sympy.Symbol('y')
z = sympy.Symbol('z')
w = sympy.Symbol('w')
xx = x * x
yy = y * y
zz = z * z
xy = x * y
xz = x * z
yz = y * z
wx = w * x
wy = w * y
wz = w * z
q = sympy.Matrix([[x], [y], [z], [w]])
R = sympy.Matrix([
    [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
    [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
    [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
])
S = sympy.Matrix([
    [sympy.Symbol("s00"), 0, 0],
    [0, sympy.Symbol("s11"), 0],
    [0, 0, sympy.Symbol("s22")]
])
M = R @ S
M_vector = sympy.Matrix([[M[i, j]] for i in range(3) for j in range(3)])
d_M_d_q = M_vector.jacobian(q)
print(d_M_d_q.shape)
print(latex(d_M_d_q))
pprint(d_M_d_q, use_unicode=True)

# %%
R = sympy.MatrixSymbol('R', 3, 3)
S = sympy.Matrix([
    [sympy.Symbol("s00"), 0, 0],
    [0, sympy.Symbol("s11"), 0],
    [0, 0, sympy.Symbol("s22")]
])
Sigma = R @ S @ S.transpose() @ R.transpose()
pprint(Sigma.as_explicit(), use_unicode=True)
# %%
R = sympy.MatrixSymbol('R', 3, 3)
S = sympy.MatrixSymbol('S', 3, 1)
Sigma = R @ S @ S.transpose() @ R.transpose()
pprint(Sigma.as_explicit(), use_unicode=True)
#
# %%
J = sympy.MatrixSymbol('J', 2, 3)
T = sympy.MatrixSymbol('T', 4, 4)
cx = sympy.Symbol('cx')
cy = sympy.Symbol('cy')
cz = sympy.Symbol('cz')
cw = sympy.Symbol('cw')

cov = sympy.Matrix([cx, cy, cz, cw])
R = sympy.Matrix([
    [1 - 2 * (cy * cy + cz * cz), 2 * (cx * cy - cw * cz),
     2 * (cx * cz + cw * cy)],
    [2 * (cx * cy + cw * cz), 1 - 2 *
     (cx * cx + cz * cz), 2 * (cy * cz - cw * cx)],
    [2 * (cx * cz - cw * cy), 2 * (cy * cz + cw * cx),
     1 - 2 * (cx * cx + cy * cy)]
])
tx = sympy.Symbol('tx')
ty = sympy.Symbol('ty')
tz = sympy.Symbol('tz')
translation = sympy.Matrix([tx, ty, tz])
sx = sympy.Symbol('sx')
sy = sympy.Symbol('sy')
sz = sympy.Symbol('sz')
s = sympy.Matrix([sx, sy, sz])
S = sympy.Matrix([
    [s[0, 0], 0, 0],
    [0, s[1, 0], 0],
    [0, 0, s[2, 0]]
])
Sigma = R @ S @ S.transpose() @ R.transpose()
homogeneous_translation_camera = T @ sympy.Matrix(
    [translation[0, 0], translation[1, 0], translation[2, 0], 1])
translation_camera = sympy.Matrix([homogeneous_translation_camera[0, 0],
                                  homogeneous_translation_camera[1, 0], homogeneous_translation_camera[2, 0]])
uv1 = (J @ translation_camera) / translation_camera[2, 0]
uv = sympy.Matrix([uv1[0, 0], uv1[1, 0]])
W = T[:3, :3]
cov_uv = J @ W @ Sigma @ W.transpose() @ J.transpose()
cov_uv_vec = sympy.Matrix(
    [cov_uv[0, 0], cov_uv[0, 1], cov_uv[1, 0], cov_uv[1, 1]])
J_cov_uv_cov = cov_uv_vec.jacobian(cov)
print(J_cov_uv_cov.shape)
print(sympy.python(J_cov_uv_cov))
J_cov_uv_s = cov_uv_vec.jacobian(s)
print(J_cov_uv_s.shape)
print(sympy.python(J_cov_uv_s))
J_uv_translation = uv.jacobian(translation)
print(J_uv_translation.shape)
print(sympy.python(J_uv_translation))
# %%
fx = sympy.Symbol('fx')
fy = sympy.Symbol('fy')
cx = sympy.Symbol('cx')
cy = sympy.Symbol('cy')
x = sympy.Symbol('x')
y = sympy.Symbol('y')
z = sympy.Symbol('z')
K = sympy.Matrix([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])
uv1 = K @ sympy.Matrix([[x], [y], [z]]) / z
uv = sympy.Matrix([uv1[0, 0], uv1[1, 0]])
J = uv.jacobian(sympy.Matrix([[x], [y], [z]]))
print(J.shape)
print(sympy.python(J))

# %%
import sympy
xy = sympy.MatrixSymbol('xy', 2, 1)
mu = sympy.Matrix(["mu_x", "mu_y"])
cov = sympy.MatrixSymbol('cov', 2, 2)
det_cov = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]
inv_cov = (1 / det_cov) * sympy.Matrix([
    [cov[1, 1], -cov[0, 1]],
    [-cov[1, 0], cov[0, 0]]
])
tmp = -0.5 * ((xy - mu).transpose()) @ inv_cov @ (xy - mu)
p = sympy.exp(tmp[0, 0])
p_vec = sympy.Matrix([p])
# %%
print(tmp.shape)
tmp1 = sympy.Matrix([tmp[0, 0]])
# %%
sympy.simplify(tmp1.jacobian(xy))
# %%
J = p_vec.jacobian(tmp)
print(J.shape)
print(sympy.python(J))
# %%
import numpy as np
xy = np.array([1, 2])
x = xy[0]
y = xy[1]
mu = np.array([3, 1])
mu_x = mu[0]
mu_y = mu[1]
cov = np.array([
    0.8, 0.1,
    0.1, 0.8]).reshape(2, 2)
det_cov = cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[1, 0]

# e = MutableDenseMatrix([[(-mu_x + xy[0, 0])*cov[1, 1]/(cov[0, 0]*cov[1, 1] - cov[0, 1]*cov[1, 0])
# - Float('0.5', precision=53)*(-mu_y + xy[1, 0])*cov[0, 1]/(cov[0, 0]*cov[1, 1] - cov[0, 1]*cov[1, 0])
# - Float('0.5', precision=53)*(-mu_y + xy[1, 0])*cov[1, 0]/(cov[0, 0]*cov[1, 1] - cov[0, 1]*cov[1, 0]))
# *exp(-Float('0.5', precision=53)*(-mu_x + xy[0, 0])*((-mu_x + xy[0, 0])*cov[1, 1]/(cov[0, 0]*cov[1, 1] - cov[0, 1]*cov[1, 0]) - (-mu_y + xy[1, 0])*cov[1, 0]/(cov[0, 0]*cov[1, 1] - cov[0, 1]*cov[1, 0])) - Float('0.5', precision=53)*(-mu_y + xy[1, 0])*(-(-mu_x + xy[0, 0])*cov[0, 1]/(cov[0, 0]*cov[1, 1] - cov[0, 1]*cov[1, 0]) + (-mu_y + xy[1, 0])*cov[0, 0]/(cov[0, 0]*cov[1, 1] - cov[0, 1]*cov[1, 0])))/(2*pi*sqrt(cov[0, 0]*cov[1, 1] - cov[0, 1]*cov[1, 0])), (-Float('0.5', precision=53)*(-mu_x + xy[0, 0])*cov[0, 1]/(cov[0, 0]*cov[1, 1] - cov[0, 1]*cov[1, 0]) - Float('0.5', precision=53)*(-mu_x + xy[0, 0])*cov[1, 0]/(cov[0, 0]*cov[1, 1] - cov[0, 1]*cov[1, 0]) + Float('1.0', precision=53)*(-mu_y + xy[1, 0])*cov[0, 0]/(cov[0, 0]*cov[1, 1] - cov[0, 1]*cov[1, 0]))*exp(-Float('0.5', precision=53)*(-mu_x + xy[0, 0])*((-mu_x + xy[0, 0])*cov[1, 1]/(cov[0, 0]*cov[1, 1] - cov[0, 1]*cov[1, 0]) - (-mu_y + xy[1, 0])*cov[1, 0]/(cov[0, 0]*cov[1, 1] - cov[0, 1]*cov[1, 0])) - Float('0.5', precision=53)*(-mu_y + xy[1, 0])*(-(-mu_x + xy[0, 0])*cov[0, 1]/(cov[0, 0]*cov[1, 1] - cov[0, 1]*cov[1, 0]) + (-mu_y + xy[1, 0])*cov[0, 0]/(cov[0, 0]*cov[1, 1] - cov[0, 1]*cov[1, 0])))/(2*pi*sqrt(cov[0, 0]*cov[1, 1] - cov[0, 1]*cov[1, 0]))]])
d_p_d_mu = np.array([
    (x-mu_x) * cov[1, 1] / det_cov - 0.5 * (y-mu_y) *
    cov[0, 1] / det_cov - 0.5 * (y-mu_y) * cov[1, 0] / det_cov,
])
print(d_p_d_mu)


def gaussian_pdf(x, mean, cov):
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    diff = x - mean
    exponent = -0.5 * diff.T @ inv_cov @ diff
    return np.exp(exponent)


def gradient_mean(x, mean, cov):
    inv_cov = np.linalg.inv(cov)
    diff = x - mean
    pdf = gaussian_pdf(x, mean, cov)
    d_pdf_d_mean = pdf * (inv_cov @ diff)
    return d_pdf_d_mean


def gradient_cov(x, mean, cov):
    inv_cov = np.linalg.inv(cov)
    diff = x - mean
    diff_outer = np.outer(diff, diff)
    pdf = gaussian_pdf(x, mean, cov)

    gradient = 0.5 * pdf * (inv_cov @ diff_outer @ inv_cov)
    return gradient


print(gradient_mean(xy, mu, cov))
print(gradient_cov(xy, mu, cov))
# %%
import torch
xy = torch.tensor([1., 2.])
mu = torch.tensor([3., 1.], requires_grad=True)
cov = torch.tensor([[0.8, 0.1], [0.1, 0.8]], requires_grad=True)
inv_cov = torch.inverse(cov)
det_cov = torch.det(cov)
diff = xy - mu
exponent = -0.5 * diff.T @ inv_cov @ diff
p = torch.exp(exponent)
p.backward()
print(mu.grad)
print(cov.grad)
# %%
xy = torch.tensor([3.5, 3.5])
mu = torch.tensor([6.9347, 0.8583], requires_grad=True)
cov = torch.tensor([[87.5116, -52.6297],
                    [-52.6296,  31.8323]], requires_grad=True)
inv_cov = torch.inverse(cov)
det_cov = torch.det(cov)
diff = xy - mu
exponent = -0.5 * diff.T @ inv_cov @ diff
normalization = 1 / (2 * np.pi * (det_cov ** 0.5))
p = normalization * torch.exp(exponent)
p *= 1.7667
p.backward()
print(p)
print(mu.grad)
print(cov.grad)

# %%
ti.init(ti.gpu, debug=True)

image = torch.rand(size=(1080, 1920, 3), dtype=torch.float32,
                   device=torch.device("cuda:0"))
count = torch.randint(low=0, high=10, size=(1080, 1920),
                      dtype=torch.int32, device=torch.device("cuda:0"))
out = torch.zeros(size=(1080, 1920), dtype=torch.float32,
                  device=torch.device("cuda:0"))


@ti.kernel
def test_taichi(
    width: ti.i32,
    height: ti.i32,
    image: ti.types.ndarray(ti.f32, ndim=3),
    count: ti.types.ndarray(ti.i32, ndim=2),
    out: ti.types.ndarray(ti.f32, ndim=2),
):
    for row, col in ti.ndrange(height, width):
        acc = 0.
        c = count[row, col]
        for i in range(c):
            if col + i < width:
                acc += image[row, col + i, 0]
                acc += image[row, col + i, 0]
                acc += image[row, col + i, 0]
        out[row, col] = acc


test_taichi(1920, 1080, image, count, out)

# %%
# tensor([0.0229, 0.9774, 0.1204, 0.1725, 0.7236, 0.7606, 0.9650, 0.1946, 0.5849],
q = np.array([0.0229, 0.9774, 0.1204, 0.1725])
# %%
np.linalg.norm(q)
# %%
# convert q to rotation matrix
R = transform.Rotation.from_quat(q)
# %%
S = np.diag([0.7606, 0.9650, 0.1946])
# %%
S
# %%
fx = 32
fy = 32
cx = 16
cy = 16
x, y, z = -0.1316, -0.2471, 1.0090
J = np.array([
    [fx / z, 0, -fx * x / (z * z)],
    [0, fy / z, -fy * y / (z * z)]])
W = np.eye(3)
# %%
Sigma = R.as_matrix() @ S @ S @ R.as_matrix().T

# %%
Sigma
# %%
cov = J @ W @ Sigma @ W.T @ J.T
# %%
cov
# %%
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])
# %%
uv1 = K @ np.array([[x], [y], [z]]) / z
# %%
uv1

# %%


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

    # Compute the quaternion outer product
    q_outer = torch.einsum('...i,...j->...ij', q, q)

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


q = torch.tensor([0.0229, 0.9774, 0.1204, 0.1725])
q = q / torch.norm(q)
print(quaternion_to_rotation_matrix_torch(q))
# %%
np_R = transform.Rotation.from_quat(q).as_matrix()
print(np_R)
# %%

T_camera_pointcloud = torch.tensor([[1., 0., 0., 0.],
                                    [0., 1., 0., 0.],
                                    [0., 0., 1., 2.],
                                    [0., 0., 0., 1.]])
camera_intrinsics = torch.tensor([[32.,  0., 16.],
                                  [0., 32., 16.],
                                  [0.,  0.,  1.]])
direction = torch.tensor([-0.3906, -0.3906,  5.0000])
origin = torch.tensor([0., 0., -2.])
xyz = torch.tensor([-0.4325, -0.7224, -0.4733])
features = torch.tensor([0.0115,  0.5507,  0.6920,  0.4666,  0.6306,  0.0871, -0.0112,  1.7667,
                         2.2963,  0.1560,  0.8710,  0.3418,  0.3658,  0.1913,  0.8727,  0.3608,
                         0.6874,  0.7516,  0.9281,  0.5649,  0.9469,  0.9090,  0.7356,  0.5436,
                         1.7886,  0.7542,  0.9568,  0.2868,  0.3552,  0.3872,  0.0827,  0.4101,
                         0.7783,  0.6266,  0.9601,  0.8252,  0.7846,  0.0183,  0.6635,  0.4688,
                         -1.4012,  0.1584,  0.3252,  0.5403,  0.4992,  0.2780,  0.7412,  0.5056,
                         0.8236,  0.9722,  0.5467,  0.6644,  0.2583,  0.0953,  0.3986,  0.2265])
q = features[:4]
s = features[4:7]
point_alpha = features[7]
r_sh = features[8:24]
g_sh = features[24:40]
b_sh = features[40:56]
pixel_uv = torch.tensor([3, 3])

torch_single_point_alpha_forward(
    point_xyz=xyz,
    point_q=q,
    point_s=s,
    T_camera_pointcloud=T_camera_pointcloud,
    camera_intrinsics=camera_intrinsics,
    point_alpha=point_alpha,
    pixel_uv=pixel_uv
)

# %%
q = torch.tensor([0.0229, 0.9774, 0.1204, 0.1725])
q = q / torch.norm(q)
R = quaternion_to_rotation_matrix_torch(q)
t = torch.rand(size=(3,))
T_pointcloud_camera = torch.eye(4)
T_pointcloud_camera[:3, :3] = R
T_pointcloud_camera[:3, 3] = t
camera_info = CameraInfo(
    camera_width=1920,
    camera_height=1080,
    camera_intrinsics=torch.tensor([[500., 0., 960.],
                                    [0., 500., 540.],
                                    [0., 0., 1.]]),
    camera_id=0)
ray_origin, ray_direction = get_ray_origin_and_direction_from_camera(
    camera_info=camera_info,
    T_pointcloud_camera=T_pointcloud_camera)

# %%
import pandas as pd
import numpy as np
path = "logs/sigmoid_on_image_fix_bug/scene_66000.parquet"
df = pd.read_parquet(path)
# %%
df.head()
# %%
import matplotlib.pyplot as plt
plt.hist(np.exp(df.cov_s0), bins=100)
# %%
np.exp(df.cov_s0).argmax()
# 44837
df.iloc[44837]
# %%
np.median(np.exp(df.cov_s0))
# %%
df.info()
# %%
# count number of NaNs in each column
for col in df.columns:
    print(col, df[col].isnull().sum())

# %%
df = df.dropna() 

# %%
import numpy as np
from sympy import symbols, Matrix, diff, exp
def compute_derivatives(mu, Sigma, x):
    # 定义符号变量
    mu1, mu2, x1, x2 = symbols('mu1 mu2 x1 x2')
    s11, s12, s21, s22 = symbols('s11 s12 s21 s22')

    # 定义向量和矩阵
    mu_sym = Matrix([mu1, mu2])
    x_sym = Matrix([x1, x2])
    Sigma_sym = Matrix([[s11, s12], [s21, s22]])

    # 定义概率密度函数
    p = exp(-1/2 * (mu_sym - x_sym).T * Sigma_sym.inv() * (mu_sym - x_sym))

    # 计算导数
    dp_dmu = diff(p[0], mu_sym)
    dp_dSigma = diff(p[0], Sigma_sym)
    """
    sympy.sympify(dp_dmu)
    sympy.sympify(dp_dSigma)
    print("dp_dmu:", dp_dmu)
    print("dp_dSigma:", dp_dSigma)
    """

    # 用实际值替换符号变量
    subs = {mu1: mu[0], mu2: mu[1], x1: x[0], x2: x[1], 
            s11: Sigma[0, 0], s12: Sigma[0, 1], s21: Sigma[1, 0], s22: Sigma[1, 1]}
    dp_dmu_val = dp_dmu.subs(subs)
    dp_dSigma_val = dp_dSigma.subs(subs)

    return dp_dmu_val, dp_dSigma_val

# 测试函数
mu = np.array([1, 2])
Sigma = np.array([[100., 0], [0, 100]])
x = np.array([0, 0])

dp_dmu, dp_dSigma = compute_derivatives(mu, Sigma, x)
print("dp/dmu:", dp_dmu)
print("dp/dSigma:", dp_dSigma)
# %%
import torch

def compute_derivatives_torch(mu, Sigma, x):
    # 将输入转换为PyTorch张量，并设置requires_grad=True以启用自动微分
    mu_torch = torch.tensor(mu, dtype=torch.float32, requires_grad=True)
    Sigma_torch = torch.tensor(Sigma, dtype=torch.float32, requires_grad=True)
    x_torch = torch.tensor(x, dtype=torch.float32)

    # 定义概率密度函数
    diff = mu_torch - x_torch
    exponent = -0.5 * diff @ torch.inverse(Sigma_torch) @ diff
    p_torch = torch.exp(exponent)

    # 计算导数
    p_torch.backward()

    return mu_torch.grad, Sigma_torch.grad

def my_compute(mu, Sigma, x):
    gaussian_mean = mu
    xy = x
    gaussian_covariance = Sigma
    xy_mean = xy - gaussian_mean
    # det_cov = gaussian_covariance.determinant()
    det_cov = Sigma[0, 0] * Sigma[1, 1] - Sigma[0, 1] * Sigma[1, 0]
    inv_cov = (1. / det_cov) * \
        np.array([[gaussian_covariance[1, 1], -gaussian_covariance[0, 1]],
                      [-gaussian_covariance[1, 0], gaussian_covariance[0, 0]]])
    cov_inv_xy_mean = inv_cov @ xy_mean
    xy_mean_T_cov_inv_xy_mean = xy_mean @ cov_inv_xy_mean
    exponent = -0.5 * xy_mean_T_cov_inv_xy_mean
    p = np.exp(exponent)
    d_p_d_mean = p * cov_inv_xy_mean
    xy_mean_outer_xy_mean = np.array([[xy_mean[0] * xy_mean[0], xy_mean[0] * xy_mean[1]],
                                      [xy_mean[1] * xy_mean[0], xy_mean[1] * xy_mean[1]]])
    d_p_d_cov = 0.5 * p * (inv_cov @
                            xy_mean_outer_xy_mean @ inv_cov)
    return d_p_d_mean, d_p_d_cov

# 测试函数
mu = np.array([1.0, 2.0])
Sigma = np.array([[1.0, 0.0], [0.0, 1.0]])
x = np.array([0.0, 0.0])

dp_dmu_torch, dp_dSigma_torch = compute_derivatives_torch(mu, Sigma, x)
dp_dmu_my, dp_dSigma_my = my_compute(mu, Sigma, x)
print("dp/dmu (torch):", dp_dmu_torch)
print("dp/dSigma (torch):", dp_dSigma_torch)
print("dp/dmu (my):", dp_dmu_my)
print("dp/dSigma (my):", dp_dSigma_my)
# %%
for i in range(100):
    mu = np.random.rand(2) * 1
    x = np.random.rand(2) * 1
    # covariance matrix must be symmetric and positive semi-definite
    tmp = np.random.rand(2, 2)
    Sigma = tmp @ tmp.T
    dp_dmu, dp_dSigma = compute_derivatives(mu, Sigma, x)
    dp_dmu, dp_dSigma = np.array(dp_dmu, dtype=np.float32), np.array(dp_dSigma, dtype=np.float32)
    dp_dmu = dp_dmu.reshape(-1)
    dp_dSigma = dp_dSigma.reshape(2, 2)
    dp_dmu_torch, dp_dSigma_torch = compute_derivatives_torch(mu, Sigma, x)
    dp_dmu_my, dp_dSigma_my = my_compute(mu, Sigma, x)

    print("dp/dmu:", dp_dmu)
    print("dp/dmu (my):", dp_dmu_my)
    print("dp/dmu (torch):", dp_dmu_torch)
    print("dp/dSigma:", dp_dSigma)
    print("dp/dSigma (my):", dp_dSigma_my)
    print("dp/dSigma (torch):", dp_dSigma_torch)
    
    # assert np.allclose(dp_dmu, dp_dmu_torch.detach().numpy(), rtol=1e-3), f"dp_dmu: {dp_dmu}, dp_dmu_torch: {dp_dmu_torch}"
    # assert np.allclose(dp_dSigma, dp_dSigma_torch.detach().numpy(), rtol=1e-3), f"dp_dSigma: {dp_dSigma}, dp_dSigma_torch: {dp_dSigma_torch}"
    assert np.allclose(dp_dmu, dp_dmu_my, rtol=1e-3), f"dp_dmu: {dp_dmu}, dp_dmu_my: {dp_dmu_my}"
    assert np.allclose(dp_dSigma, dp_dSigma_my, rtol=1e-3), f"dp_dSigma: {dp_dSigma}, dp_dSigma_my: {dp_dSigma_my}"
    
    
# %%
import pandas as pd
import numpy as np
import open3d as o3d
parquet_path = "/home/kuangyuan/hdd/Development/taichi_3d_gaussian_splatting/logs/tat_truck_experiment_more_val/scene_13750.parquet"
df = pd.read_parquet(parquet_path)
# %%
df.head()
# %%
point_cloud = df[["x", "y", "z"]].values
point_cloud_rgb = df[["r_sh0", "g_sh0", "b_sh0"]].values
# here rgb are actually sh coefficients (-inf, inf), 
# need to apply sigmoid to get (0, 1) rgb
point_cloud_rgb = 1.0 / (1.0 + np.exp(-point_cloud_rgb))
# %%
point_cloud_o3d = o3d.geometry.PointCloud()
point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud)
point_cloud_o3d.colors = o3d.utility.Vector3dVector(point_cloud_rgb)
# visualize point cloud
o3d.visualization.draw_geometries([point_cloud_o3d])

# %%
# then only visualize the points close to the origin
mask = np.linalg.norm(point_cloud, axis=1) < 10
point_cloud_o3d = o3d.geometry.PointCloud()
point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud[mask])
point_cloud_o3d.colors = o3d.utility.Vector3dVector(point_cloud_rgb[mask])
o3d.visualization.draw_geometries([point_cloud_o3d])
# %%
q = df[["cov_q0", "cov_q1", "cov_q2", "cov_q3"]].values
s = df[["cov_s0", "cov_s1", "cov_s2"]].values

# each point is actually a ellipsoid with rotation q and scale s
# here we use the shortest axis as the direction of the normal
# and the length of the longest axis as the length of the normal to visualize
# first, create a [0, 0, 1] like vector with shape (N, 3), the shortest axis is 1
# then, rotate the vector by q, the shortest axis is rotated to the direction of the normal
base_vector = np.zeros((len(q), 3))
shortest_axis = np.argmin(s, axis=1)
base_vector[np.arange(len(q)), shortest_axis] = 1.0
# quaternion rotation, q is xyzw
"""
def rotation_matrix_from_quaternion(q: ti.math.vec4) -> ti.math.mat3:
    xx = q.x * q.x
    yy = q.y * q.y
    zz = q.z * q.z
    xy = q.x * q.y
    xz = q.x * q.z
    yz = q.y * q.z
    wx = q.w * q.x
    wy = q.w * q.y
    wz = q.w * q.z
    return ti.math.mat3([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ])
"""

def rotation_matrix_from_quaternion(q: np.ndarray) -> np.ndarray:
    xx = q[0] * q[0]
    yy = q[1] * q[1]
    zz = q[2] * q[2]
    xy = q[0] * q[1]
    xz = q[0] * q[2]
    yz = q[1] * q[2]
    wx = q[3] * q[0]
    wy = q[3] * q[1]
    wz = q[3] * q[2]
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ])

S = np.exp(s)

rotated_S = np.zeros((len(q), 3))
normal = np.zeros((len(q), 3))
for i in range(len(q)):
    rotated_S[i] = rotation_matrix_from_quaternion(q[i]) @ S[i]
    normal[i] = rotation_matrix_from_quaternion(q[i]) @ base_vector[i]
    normal[i] *= np.linalg.norm(rotated_S[i])

 
    
# %%
point_cloud_o3d = o3d.geometry.PointCloud()
point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud[mask])
point_cloud_o3d.colors = o3d.utility.Vector3dVector(point_cloud_rgb[mask])
point_cloud_o3d.normals = o3d.utility.Vector3dVector(normal[mask] * 3)
o3d.visualization.draw_geometries([point_cloud_o3d])
# %%

import taichi as ti
ti.init(arch=ti.cpu)
@ti.kernel
def test():
    Cov = ti.Matrix([
        [0.5, 0.2],
        [0.2, 0.7]
    ])
    eig, V = ti.sym_eig(Cov)
    print(eig)
    print(V)
test()
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define the mean and covariance matrix
mean = np.array([0, 0])
cov = np.array([[50, 30], [30, 50]])  # Replace with your covariance matrix
alpha = 0.6

eigen = np.linalg.eig(cov)
eigen_values = eigen[0]
eigen_vectors = eigen[1]

# Define the threshold
threshold = 1 / 255.

# Create a grid of points
x = np.linspace(-50, 50, 100)
y = np.linspace(-50, 50, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# Create a 2D Gaussian distribution
rv = multivariate_normal(mean, cov)
# Create a unnormalized 2D Gaussian distribution
normalize_factor = 1 / (2 * np.pi * np.sqrt(np.linalg.det(cov)))

# Evaluate the PDF at each point in the grid
pdf_values = rv.pdf(pos) / normalize_factor * alpha

# Create a binary mask where the PDF is greater than the threshold
mask = pdf_values > threshold

# Plot the area where the PDF is greater than the threshold
plt.figure(figsize=(6, 6))
plt.imshow(mask, extent=(-50, 50, -50, 50), origin='lower')

# plt eigenvectors
plt.quiver(mean[0], mean[1], eigen_vectors[0, 0], eigen_vectors[1, 0], color='r', scale=10 / np.sqrt(eigen_values[0]))
plt.quiver(mean[0], mean[1], eigen_vectors[0, 1], eigen_vectors[1, 1], color='r', scale=10 / np.sqrt(eigen_values[1]))

plt.colorbar()
plt.show()
# %%
print(np.sqrt(eigen_values[0]) * 4)
print(np.sqrt(eigen_values[1]) * 4)

# %%