import unittest
import taichi as ti
import numpy as np
import scipy.spatial.transform as transform
from taichi_3d_gaussian_splatting.GaussianPoint3D import GaussianPoint3D, rotation_matrix_from_quaternion


class GaussianPoint3d_test(unittest.TestCase):
    def setUp(self) -> None:
        ti.init(ti.gpu, debug=True)

    def test_project_to_camera_covariance(self):
        T_camera_world = np.eye(4, dtype=np.float32)
        projective_transform = np.array(
            [[32, 0, 16], [0, 32, 16], [0, 0, 1]], dtype=np.float32)
        xyz = np.array([-0.1316, -0.2471, 1.0090], dtype=np.float32)
        s = np.array([np.log(0.7606), np.log(0.9650), np.log(0.1946)])
        q = np.array([0.0229, 0.9774, 0.1204, 0.1725])
        R = transform.Rotation.from_quat(q)
        S = np.diag(np.exp(s))
        fx, fy, cx, cy = projective_transform[0, 0], projective_transform[1,
                                                                          1], projective_transform[0, 2], projective_transform[1, 2]
        x, y, z = xyz
        J = np.array([
            [fx / z, 0, -fx * x / (z * z)],
            [0, fy / z, -fy * y / (z * z)]])
        W = np.eye(3)
        Sigma = R.as_matrix() @ S @ S @ R.as_matrix().T
        cov = J @ W @ Sigma @ W.T @ J.T
        T_camera_world_ti = ti.math.mat4.field(shape=())
        T_camera_world_ti.from_numpy(T_camera_world)
        projective_transform_ti = ti.math.mat3.field(shape=())
        projective_transform_ti.from_numpy(projective_transform)
        xyz_ti = ti.math.vec3.field(shape=())
        xyz_ti.from_numpy(xyz)
        s_ti = ti.math.vec3.field(shape=())
        s_ti.from_numpy(s)
        q_ti = ti.math.vec4.field(shape=())
        q_ti.from_numpy(q)

        @ti.kernel
        def call_project_to_camera_covariance() -> ti.math.mat2:
            p = GaussianPoint3D(
                cov_rotation=q_ti[None],
                cov_scale=s_ti[None],
            )
            return p.project_to_camera_covariance(
                T_camera_world_ti[None],
                projective_transform_ti[None],
                xyz_ti[None]
            )
        cov_ti = call_project_to_camera_covariance()
        self.assertTrue(np.allclose(cov_ti.to_numpy(), cov, rtol=1e-2),
                        msg=f"Expected: {cov}, Actual: {cov_ti.to_numpy()}")

    def test_rotation_matrix_from_quaternion(self):
        q = np.array([0.0229, 0.9774, 0.1204, 0.1725])
        np_R = transform.Rotation.from_quat(q).as_matrix()
        q_ti = ti.math.vec4.field(shape=())
        q_ti.from_numpy(q)

        @ti.kernel
        def call_rotation_matrix_from_quaternion() -> ti.math.mat3:
            return rotation_matrix_from_quaternion(q_ti[None])
        R = call_rotation_matrix_from_quaternion()
        self.assertTrue(np.allclose(R.to_numpy(), np_R, atol=1e-2),
                        msg=f"Expected: {np_R}, Actual: {R.to_numpy()}")
