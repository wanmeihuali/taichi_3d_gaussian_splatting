# %%
import taichi as ti
import taichi.math
from .SphericalHarmonics import SphericalHarmonics, vec16f
from .utils import ti_sigmoid, ti_sigmoid_with_jacobian, quaternion_rotate

mat2x3f = ti.types.matrix(n=2, m=3, dtype=ti.f32)
mat9x9f = ti.types.matrix(n=9, m=9, dtype=ti.f32)
mat9x3f = ti.types.matrix(n=9, m=3, dtype=ti.f32)
mat4x9f = ti.types.matrix(n=4, m=9, dtype=ti.f32)
mat9x4f = ti.types.matrix(n=9, m=4, dtype=ti.f32)


@ti.func
def project_point_to_camera(
    translation: ti.math.vec3,
    T_camera_world: ti.math.mat4,
    projective_transform: ti.math.mat3,
):
    homogeneous_translation_camera = T_camera_world @ ti.math.vec4(
        translation.x, translation.y, translation.z, 1)
    translation_camera = ti.math.vec3(
        homogeneous_translation_camera.x, homogeneous_translation_camera.y, homogeneous_translation_camera.z)
    uv1 = (projective_transform @ translation_camera) / \
        translation_camera.z
    uv = ti.math.vec2(uv1.x, uv1.y)
    return uv, translation_camera


@ti.func
def rotation_matrix_from_quaternion(q: ti.math.vec4) -> ti.math.mat3:
    """
    Convert a quaternion to a rotation matrix.
    """
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


@ti.func
def transform_matrix_from_quaternion_and_translation(q: ti.math.vec4, t: ti.math.vec3) -> ti.math.mat4:
    """
    Convert a quaternion and a translation to a transformation matrix.
    """
    R = rotation_matrix_from_quaternion(q)
    return ti.math.mat4([
        [R[0, 0], R[0, 1], R[0, 2], t[0]],
        [R[1, 0], R[1, 1], R[1, 2], t[1]],
        [R[2, 0], R[2, 1], R[2, 2], t[2]],
        [0, 0, 0, 1]
    ])


@ti.func
def get_projective_transform_jacobian(
    projective_transform: ti.math.mat3,
    xyz: ti.math.vec3,
):
    # [[fx/z, 0, cx/z - (cx*z + fx*x)/z**2], [0, fy/z, cy/z - (cy*z + fy*y)/z**2]]
    fx = projective_transform[0, 0]
    fy = projective_transform[1, 1]
    cx = projective_transform[0, 2]
    cy = projective_transform[1, 2]
    x = xyz.x
    y = xyz.y
    z = xyz.z
    """
    return mat2x3f([
        [fx/z, 0, cx/z],
        [0, fy/z, cy/z]
    ])
    """
    return mat2x3f([
        [fx / z, 0, -(fx * x) / (z * z)],
        [0, fy / z, -(fy * y) / (z * z)]
    ])


@ti.func
def box_muller_transform(u1, u2):
    z1 = ti.sqrt(-2 * ti.log(u1)) * ti.cos(2 * 3.141592653589 * u2)
    z2 = ti.sqrt(-2 * ti.log(u1)) * ti.sin(2 * 3.141592653589 * u2)
    return z1, z2


@ti.dataclass
class GaussianPoint3D:
    """
    As the paper said: The covariance matrix Σ of a 3D Gaussian is analogous to describing the configuration of an ellipsoid. 
    """
    cov_rotation: ti.math.vec4  # quaternion of cov_rotation, x, y, z, w
    cov_scale: ti.math.vec3  # cov_scale of x, y, z
    translation: ti.math.vec3  # translation of x, y, z
    alpha: ti.f32  # opacity of the point
    color_r: vec16f  # color of the point, r
    color_g: vec16f  # color of the point, g
    color_b: vec16f  # color of the point, b

    @ti.func
    def project_to_camera_position(
        self,
        T_camera_world: ti.math.mat4,
        projective_transform: ti.math.mat3,
    ):
        return project_point_to_camera(self.translation, T_camera_world, projective_transform)

    @ti.func
    def project_to_camera_position_with_extra_translation_and_rotation_and_scale(
        self,
        T_camera_world: ti.math.mat4,
        projective_transform: ti.math.mat3,
        extra_translation: ti.math.vec3,
        extra_rotation: ti.math.vec4,
        extra_scale: ti.math.vec3,
    ):
        translation = self.translation * extra_scale + extra_translation
        # extra_rotation is xyzw quaternion
        translation = quaternion_rotate(extra_rotation, translation)
        return project_point_to_camera(translation, T_camera_world, projective_transform)

    @ti.func
    def project_to_camera_position_jacobian(
        self,
        T_camera_world: ti.math.mat4,
        projective_transform: ti.math.mat3,
    ):
        T = T_camera_world
        W = ti.math.mat3([
            [T[0, 0], T[0, 1], T[0, 2]],
            [T[1, 0], T[1, 1], T[1, 2]],
            [T[2, 0], T[2, 1], T[2, 2]]
        ])

        # the forward output is uv, cov_uv, the backward input is d_uv, d_cov_uv
        # jacobian: d_uv / d_translation, d_cov_uv / d_cov_rotation, d_cov_uv / d_cov_scale
        # in paper, d_cov_uv / d_cov_rotation is called d_Sigma' / dq, d_cov_uv / d_cov_scale is called d_Sigma' / ds
        # t = self.translation
        t = T_camera_world @ ti.math.vec4(
            [self.translation.x, self.translation.y, self.translation.z, 1])
        K = projective_transform
        # d_uv_d_translation_camera = \left[\begin{matrix}\frac{K_{0, 0}}{t_{2, 0}} & \frac{K_{0, 1}}{t_{2, 0}} & \frac{- K_{0, 0} t_{0, 0} - K_{0, 1} t_{1, 0}}{t_{2, 0}^{2}}\\\frac{K_{1, 0}}{t_{2, 0}} & \frac{K_{1, 1}}{t_{2, 0}} & \frac{- K_{1, 0} t_{0, 0} - K_{1, 1} t_{1, 0}}{t_{2, 0}^{2}}\end{matrix}\right]
        d_uv_d_translation_camera = mat2x3f([
            [K[0, 0] / t.z, K[0, 1] / t.z,
                (-K[0, 0] * t.x - K[0, 1] * t.y) / (t.z * t.z)],
            [K[1, 0] / t.z, K[1, 1] / t.z, (-K[1, 0] * t.x - K[1, 1] * t.y) / (t.z * t.z)]])
        d_translation_camera_d_translation = W
        d_uv_d_translation = d_uv_d_translation_camera @ d_translation_camera_d_translation  # 2 x 3
        return d_uv_d_translation

    @ti.func
    def project_to_camera_covariance(
        self,
        T_camera_world: ti.math.mat4,
        projective_transform: ti.math.mat3,
        translation_camera: ti.math.vec3,
    ):
        """
        Project the Gaussian point to camera space, without jacobian.
        """
        J = get_projective_transform_jacobian(
            projective_transform, translation_camera)
        T = T_camera_world
        R = rotation_matrix_from_quaternion(self.cov_rotation)
        exp_cov_scale = ti.math.exp(self.cov_scale)
        S = ti.math.mat3([
            [exp_cov_scale.x, 0, 0],
            [0, exp_cov_scale.y, 0],
            [0, 0, exp_cov_scale.z]
        ])
        # covariance matrix, 3x3, equation (6) in the paper
        Sigma = R @ S @ S.transpose() @ R.transpose()

        W = ti.math.mat3([
            [T[0, 0], T[0, 1], T[0, 2]],
            [T[1, 0], T[1, 1], T[1, 2]],
            [T[2, 0], T[2, 1], T[2, 2]]
        ])

        cov_uv = J @ W @ Sigma @ W.transpose() @ J.transpose()  # equation (5) in the paper
        return cov_uv

    @ti.func
    def project_to_camera_covariance_with_extra_rotation_and_scale(
        self,
        T_camera_world: ti.math.mat4,
        projective_transform: ti.math.mat3,
        translation_camera: ti.math.vec3,
        extra_rotation_quaternion: ti.math.vec4,
        extra_scale: ti.math.vec3,
    ):
        """
        Project the Gaussian point to camera space, without jacobian.
        """
        J = get_projective_transform_jacobian(
            projective_transform, translation_camera)
        T = T_camera_world
        R = rotation_matrix_from_quaternion(self.cov_rotation)
        exp_cov_scale = ti.math.exp(self.cov_scale)
        S = ti.math.mat3([
            [exp_cov_scale.x, 0, 0],
            [0, exp_cov_scale.y, 0],
            [0, 0, exp_cov_scale.z]
        ])
        # covariance matrix, 3x3, equation (6) in the paper
        Sigma = R @ S @ S.transpose() @ R.transpose()

        # for inference, we can add extra rotation and scale to the covariance matrix
        # e.g. when we want to rotate or resize point cloud for an object in the scene
        R_extra = rotation_matrix_from_quaternion(extra_rotation_quaternion)
        S_extra = ti.math.mat3([
            [extra_scale.x, 0, 0],
            [0, extra_scale.y, 0],
            [0, 0, extra_scale.z]
        ])
        Sigma = R_extra @ S_extra @ Sigma @ S_extra.transpose() @ R_extra.transpose()

        W = ti.math.mat3([
            [T[0, 0], T[0, 1], T[0, 2]],
            [T[1, 0], T[1, 1], T[1, 2]],
            [T[2, 0], T[2, 1], T[2, 2]]
        ])

        cov_uv = J @ W @ Sigma @ W.transpose() @ J.transpose()  # equation (5) in the paper
        return cov_uv

    @ti.func
    def project_to_camera_covariance_jacobian(
        self,
        T_camera_world: ti.math.mat4,
        projective_transform: ti.math.mat3,
        translation_camera: ti.math.vec3,
    ):
        """
        Project the Gaussian point to camera space, with jacobian.
        """
        J = get_projective_transform_jacobian(
            projective_transform, translation_camera)
        T = T_camera_world
        R = rotation_matrix_from_quaternion(self.cov_rotation)
        exp_cov_scale = ti.math.exp(self.cov_scale)
        S = ti.math.mat3([
            [exp_cov_scale.x, 0, 0],
            [0, exp_cov_scale.y, 0],
            [0, 0, exp_cov_scale.z]
        ])
        M = R @ S
        W = ti.math.mat3([
            [T[0, 0], T[0, 1], T[0, 2]],
            [T[1, 0], T[1, 1], T[1, 2]],
            [T[2, 0], T[2, 1], T[2, 2]]
        ])

        U = J @ W  # 2x3
        # Sigma' = U @ Sigma @ U^T, so
        # d_Sigma' / d_Sigma =
        # \left[\begin{matrix}U_{0, 0}^{2} & U_{0, 0} U_{0, 1} & U_{0, 0} U_{0, 2} & U_{0, 0} U_{0, 1} & U_{0, 1}^{2} & U_{0, 1} U_{0, 2} & U_{0, 0} U_{0, 2} & U_{0, 1} U_{0, 2} & U_{0, 2}^{2}\\U_{0, 0} U_{1, 0} & U_{0, 0} U_{1, 1} & U_{0, 0} U_{1, 2} & U_{0, 1} U_{1, 0} & U_{0, 1} U_{1, 1} & U_{0, 1} U_{1, 2} & U_{0, 2} U_{1, 0} & U_{0, 2} U_{1, 1} & U_{0, 2} U_{1, 2}\\U_{0, 0} U_{1, 0} & U_{0, 1} U_{1, 0} & U_{0, 2} U_{1, 0} & U_{0, 0} U_{1, 1} & U_{0, 1} U_{1, 1} & U_{0, 2} U_{1, 1} & U_{0, 0} U_{1, 2} & U_{0, 1} U_{1, 2} & U_{0, 2} U_{1, 2}\\U_{1, 0}^{2} & U_{1, 0} U_{1, 1} & U_{1, 0} U_{1, 2} & U_{1, 0} U_{1, 1} & U_{1, 1}^{2} & U_{1, 1} U_{1, 2} & U_{1, 0} U_{1, 2} & U_{1, 1} U_{1, 2} & U_{1, 2}^{2}\end{matrix}\right]
        # as Sigma' and Sigma are matrix and the derivative is a tensor, so we use a 2x2x3x3 tensor to represent the derivative
        # However, taichi does not support tensor well, so we flatten the tensor to a matrix 4x9
        d_Sigma_prime_d_Sigma = mat4x9f([
            [U[0, 0] * U[0, 0], U[0, 0] * U[0, 1], U[0, 0] * U[0, 2], U[0, 0] * U[0, 1], U[0, 1] *
                U[0, 1], U[0, 1] * U[0, 2], U[0, 0] * U[0, 2], U[0, 1] * U[0, 2], U[0, 2] * U[0, 2]],
            [U[0, 0] * U[1, 0], U[0, 0] * U[1, 1], U[0, 0] * U[1, 2], U[0, 1] * U[1, 0], U[0, 1] *
                U[1, 1], U[0, 1] * U[1, 2], U[0, 2] * U[1, 0], U[0, 2] * U[1, 1], U[0, 2] * U[1, 2]],
            [U[0, 0] * U[1, 0], U[0, 1] * U[1, 0], U[0, 2] * U[1, 0], U[0, 0] * U[1, 1], U[0, 1] *
                U[1, 1], U[0, 2] * U[1, 1], U[0, 0] * U[1, 2], U[0, 1] * U[1, 2], U[0, 2] * U[1, 2]],
            [U[1, 0] * U[1, 0], U[1, 0] * U[1, 1], U[1, 0] * U[1, 2], U[1, 0] * U[1, 1], U[1, 1]
                * U[1, 1], U[1, 1] * U[1, 2], U[1, 0] * U[1, 2], U[1, 1] * U[1, 2], U[1, 2] * U[1, 2]]
        ])
        # d_Sigma_d_M is 3x3x3x3 tensor, we flatten it to a matrix 9x9
        # \left[\begin{matrix}2 m_{00} & 2 m_{01} & 2 m_{02} & 0 & 0 & 0 & 0 & 0 & 0\\m_{10} & m_{11} & m_{12} & m_{00} & m_{01} & m_{02} & 0 & 0 & 0\\m_{20} & m_{21} & m_{22} & 0 & 0 & 0 & m_{00} & m_{01} & m_{02}\\m_{10} & m_{11} & m_{12} & m_{00} & m_{01} & m_{02} & 0 & 0 & 0\\0 & 0 & 0 & 2 m_{10} & 2 m_{11} & 2 m_{12} & 0 & 0 & 0\\0 & 0 & 0 & m_{20} & m_{21} & m_{22} & m_{10} & m_{11} & m_{12}\\m_{20} & m_{21} & m_{22} & 0 & 0 & 0 & m_{00} & m_{01} & m_{02}\\0 & 0 & 0 & m_{20} & m_{21} & m_{22} & m_{10} & m_{11} & m_{12}\\0 & 0 & 0 & 0 & 0 & 0 & 2 m_{20} & 2 m_{21} & 2 m_{22}\end{matrix}\right]
        d_Sigma_d_M = mat9x9f([
            [2 * M[0, 0], 2 * M[0, 1], 2 * M[0, 2], 0, 0, 0, 0, 0, 0],
            [M[1, 0], M[1, 1], M[1, 2], M[0, 0], M[0, 1], M[0, 2], 0, 0, 0],
            [M[2, 0], M[2, 1], M[2, 2], 0, 0, 0, M[0, 0], M[0, 1], M[0, 2]],
            [M[1, 0], M[1, 1], M[1, 2], M[0, 0], M[0, 1], M[0, 2], 0, 0, 0],
            [0, 0, 0, 2 * M[1, 0], 2 * M[1, 1], 2 * M[1, 2], 0, 0, 0],
            [0, 0, 0, M[2, 0], M[2, 1], M[2, 2], M[1, 0], M[1, 1], M[1, 2]],
            [M[2, 0], M[2, 1], M[2, 2], 0, 0, 0, M[0, 0], M[0, 1], M[0, 2]],
            [0, 0, 0, M[2, 0], M[2, 1], M[2, 2], M[1, 0], M[1, 1], M[1, 2]],
            [0, 0, 0, 0, 0, 0, 2 * M[2, 0], 2 * M[2, 1], 2 * M[2, 2]]
        ])

        d_Sigma_prime_d_M = d_Sigma_prime_d_Sigma @ d_Sigma_d_M  # 4x9
        # M = R @ S, so d_M / d_S = R
        # \left[\begin{matrix}R_{0, 0} & 0 & 0\\0 & R_{0, 1} & 0\\0 & 0 & R_{0, 2}\\R_{1, 0} & 0 & 0\\0 & R_{1, 1} & 0\\0 & 0 & R_{1, 2}\\R_{2, 0} & 0 & 0\\0 & R_{2, 1} & 0\\0 & 0 & R_{2, 2}\end{matrix}\right]
        d_M_d_S = mat9x3f([
            [R[0, 0], 0, 0],
            [0, R[0, 1], 0],
            [0, 0, R[0, 2]],
            [R[1, 0], 0, 0],
            [0, R[1, 1], 0],
            [0, 0, R[1, 2]],
            [R[2, 0], 0, 0],
            [0, R[2, 1], 0],
            [0, 0, R[2, 2]],
        ])
        d_S_d_s = ti.math.mat3([
            [exp_cov_scale.x, 0, 0],
            [0, exp_cov_scale.y, 0],
            [0, 0, exp_cov_scale.z]
        ])
        d_Sigma_prime_d_s = d_Sigma_prime_d_M @ d_M_d_S  @ d_S_d_s  # 4x3

        # d_M / dq is 4x3x3 tensor, we flatten it to a matrix 9 x 4
        # \left[\begin{matrix}0 & - 4 s_{00} y & - 4 s_{00} z & 0\\2 s_{11} y & 2 s_{11} x & - 2 s_{11} w & - 2 s_{11} z\\2 s_{22} z & 2 s_{22} w & 2 s_{22} x & 2 s_{22} y\\2 s_{00} y & 2 s_{00} x & 2 s_{00} w & 2 s_{00} z\\- 4 s_{11} x & 0 & - 4 s_{11} z & 0\\- 2 s_{22} w & 2 s_{22} z & 2 s_{22} y & - 2 s_{22} x\\2 s_{00} z & - 2 s_{00} w & 2 s_{00} x & - 2 s_{00} y\\2 s_{11} w & 2 s_{11} z & 2 s_{11} y & 2 s_{11} x\\- 4 s_{22} x & - 4 s_{22} y & 0 & 0\end{matrix}\right]
        s = exp_cov_scale
        q = self.cov_rotation
        d_M_d_q = mat9x4f([
            [0, -4 * s.x * q.y, -4 * s.x * q.z, 0],
            [2 * s.y * q.y, 2 * s.y * q.x, -2 * s.y * q.w, -2 * s.y * q.z],
            [2 * s.z * q.z, 2 * s.z * q.w, 2 * s.z * q.x, 2 * s.z * q.y],
            [2 * s.x * q.y, 2 * s.x * q.x, 2 * s.x * q.w, 2 * s.x * q.z],
            [-4 * s.y * q.x, 0, -4 * s.y * q.z, 0],
            [-2 * s.z * q.w, 2 * s.z * q.z, 2 * s.z * q.y, -2 * s.z * q.x],
            [2 * s.x * q.z, -2 * s.x * q.w, 2 * s.x * q.x, -2 * s.x * q.y],
            [2 * s.y * q.w, 2 * s.y * q.z, 2 * s.y * q.y, 2 * s.y * q.x],
            [-4 * s.z * q.x, -4 * s.z * q.y, 0, 0]
        ])
        d_Sigma_prime_d_q = d_Sigma_prime_d_M @ d_M_d_q  # 4x4
        return d_Sigma_prime_d_q, d_Sigma_prime_d_s

    @ti.func
    def get_color_by_ray(
        self,
        ray_origin: ti.math.vec3,
        ray_direction: ti.math.vec3,
    ) -> ti.math.vec3:
        o = ray_origin
        d = ray_direction
        # TODO: try other methods to get the query point for SH, e.g. the intersection point of the ray and the ellipsoid
        r = SphericalHarmonics(self.color_r).evaluate(d)
        r_normalized = ti_sigmoid(r)
        g = SphericalHarmonics(self.color_g).evaluate(d)
        g_normalized = ti_sigmoid(g)
        b = SphericalHarmonics(self.color_b).evaluate(d)
        b_normalized = ti_sigmoid(b)
        # return ti.math.vec3(r, g, b)
        return ti.math.vec3(r_normalized, g_normalized, b_normalized)

    @ti.func
    def get_color_with_jacobian_by_ray(
        self,
        ray_origin: ti.math.vec3,
        ray_direction: ti.math.vec3,
    ):
        o = ray_origin
        d = ray_direction
        r, r_jacobian = SphericalHarmonics(
            self.color_r).evaluate_with_jacobian(d)
        r_normalized, r_normalized_jacobian = ti_sigmoid_with_jacobian(r)
        g, g_jacobian = SphericalHarmonics(
            self.color_g).evaluate_with_jacobian(d)
        g_normalized, g_normalized_jacobian = ti_sigmoid_with_jacobian(g)
        b, b_jacobian = SphericalHarmonics(
            self.color_b).evaluate_with_jacobian(d)
        b_normalized, b_normalized_jacobian = ti_sigmoid_with_jacobian(b)
        r_jacobian = r_normalized_jacobian * r_jacobian
        g_jacobian = g_normalized_jacobian * g_jacobian
        b_jacobian = b_normalized_jacobian * b_jacobian

        # return ti.math.vec3(r, g, b), r_jacobian, g_jacobian, b_jacobian
        return ti.math.vec3(r_normalized, g_normalized, b_normalized), r_jacobian, g_jacobian, b_jacobian

    @ti.func
    def get_ellipsoid_foci_vector(self) -> ti.math.vec3:
        base_vector = ti.math.vec3(1, 0, 0)
        if self.cov_scale.x < self.cov_scale.y and self.cov_scale.y > self.cov_scale.z:
            base_vector = ti.math.vec3(0, 1, 0)
        elif self.cov_scale.x < self.cov_scale.z and self.cov_scale.y < self.cov_scale.z:
            base_vector = ti.math.vec3(0, 0, 1)
        R = rotation_matrix_from_quaternion(self.cov_rotation)
        base_vector = R @ base_vector
        s = ti.exp(self.cov_scale)
        r_c = ti.max(s.x, s.y, s.z)
        r_a = ti.min(s.x, s.y, s.z)
        foci_vector = ti.sqrt(r_c**2 - r_a**2) * base_vector
        return foci_vector

    @ti.func
    def sample(self) -> ti.math.vec3:
        u1 = ti.random()
        u2 = ti.random()
        u3 = ti.random()
        u4 = ti.random()
        z1, z2 = box_muller_transform(u1, u2)
        z3, _ = box_muller_transform(u3, u4)
        R = rotation_matrix_from_quaternion(self.cov_rotation)
        exp_cov_scale = ti.math.exp(self.cov_scale)
        S = ti.math.mat3([
            [exp_cov_scale.x, 0, 0],
            [0, exp_cov_scale.y, 0],
            [0, 0, exp_cov_scale.z]
        ])
        base = ti.math.vec3(z1, z2, z3)
        return self.translation + R @ S @ base


# %%
"""
ti.init(ti.cpu)

a = GaussianPoint3D.field(shape=())


@ti.kernel
def test():
    a[None] = GaussianPoint3D(
        cov_rotation=ti.math.vec4(0, 0, 0, 1),
        cov_scale=ti.math.vec3(1, 2, 3),
        translation=ti.math.vec3(0, 1, 1))

    t = ti.math.mat4([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    p = ti.math.mat3([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
    uv, translation_camera = a[None].project_to_camera_position(t, p)
    d_uv_d_translation = a[None].project_to_camera_position_jacobian(t, p)
    print(uv)
    print(d_uv_d_translation)
    cov_uv = a[None].project_to_camera_covariance(t, p, translation_camera)
    print(cov_uv)
    d_Sigma_prime_d_q, d_Sigma_prime_d_s = a[None].project_to_camera_covariance_jacobian(
        t, p, translation_camera)
    print(d_Sigma_prime_d_q)
    print(d_Sigma_prime_d_s)


test()
"""
