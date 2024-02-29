# %%
import taichi as ti
import taichi.math as tm
# %%
vec5f = ti.types.vector(5, float)
vec7f = ti.types.vector(7, float)
vec16f = ti.types.vector(16, float)


@ti.func
def get_spherical_harmonic_from_xyz(
    xyz: tm.vec3
):
    xyz = tm.normalize(xyz)
    x, y, z = xyz.x, xyz.y, xyz.z
    l0m0 = 0.28209479177387814
    l1m1 = -0.48860251190291987 * y
    l1m0 = 0.48860251190291987 * z
    l1p1 = -0.48860251190291987 * x
    l2m2 = 1.0925484305920792 * x * y
    l2m1 = -1.0925484305920792 * y * z
    l2m0 = 0.94617469575755997 * z * z - 0.31539156525251999
    l2p1 = -1.0925484305920792 * x * z
    l2p2 = 0.54627421529603959 * x * x - 0.54627421529603959 * y * y
    l3m3 = 0.59004358992664352 * y * (-3.0 * x * x + y * y)
    l3m2 = 2.8906114426405538 * x * y * z
    l3m1 = 0.45704579946446572 * y * (1.0 - 5.0 * z * z)
    l3m0 = 0.3731763325901154 * z * (5.0 * z * z - 3.0)
    l3p1 = 0.45704579946446572 * x * (1.0 - 5.0 * z * z)
    l3p2 = 1.4453057213202769 * z * (x * x - y * y)
    l3p3 = 0.59004358992664352 * x * (-x * x + 3.0 * y * y)
    return vec16f(l0m0, l1m1, l1m0, l1p1, l2m2, l2m1, l2m0, l2p1, l2p2, l3m3, l3m2, l3m1, l3m0, l3p1, l3p2, l3p3)


@ti.dataclass
class SphericalHarmonics:
    factor: vec16f

    @ti.func
    def evaluate(
        self,
        xyz: tm.vec3
    ) -> ti.float32:
        spherical_harmonic = get_spherical_harmonic_from_xyz(xyz)
        return tm.dot(self.factor, spherical_harmonic)

    @ti.func
    def evaluate_with_jacobian(
        self,
        xyz: tm.vec3
    ):
        spherical_harmonic = get_spherical_harmonic_from_xyz(xyz)
        return tm.dot(self.factor, spherical_harmonic), spherical_harmonic
