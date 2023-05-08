import taichi as ti
import taichi.math as tm


@ti.func
def intersect_ray_with_ellipsoid(
    ray_origin: tm.vec3,
    ray_direction: tm.vec3,
    ellipsoid_R: tm.mat3,  # R
    ellipsoid_t: tm.vec3,
    ellipsoid_S: tm.vec3,
    eps: ti.f32 = 1e-5
):
    """ intersect a ray with an ellipsoid

    Args:
        ray_origin (tm.vec3): the origin of the ray in the world space
        ray_direction (tm.vec3): the direction of the ray in the world space
        ellipsoid_R (tm.mat3): the rotation matrix of the ellipsoid
        ellipsoid_S (tm.vec3): the scale of the ellipsoid
        eps (ti.f32, optional): _description_. Defaults to 1e-5.

    Returns:
        (ti.i32, tm.vec3): whether the ray intersects with the ellipsoid, and the intersection point in the world space
    """
    o = ray_origin
    d = ray_direction
    t = ellipsoid_t
    R = ellipsoid_R
    S = ellipsoid_S
    has_intersection = False
    intersection_point = tm.vec3(0.0, 0.0, 0.0)

    inv_transform_matrix = tm.mat3([
        [1 / S[0], 0, 0],
        [0, 1 / S[1], 0],
        [0, 0, 1 / S[2]]
    ]) @ (R.transpose())
    o_transformed = inv_transform_matrix @ (o - t)
    d_transformed = inv_transform_matrix @ d

    A = tm.dot(d_transformed, d_transformed)
    if abs(A) < eps:
        A = eps

    B = 2 * tm.dot(o_transformed, d_transformed)
    C = tm.dot(o_transformed, o_transformed) - 1

    discriminant = B ** 2 - 4 * A * C
    if discriminant < 0:
        has_intersection = False
    else:
        if abs(discriminant) < eps:
            discriminant = 0

        sqrt_discriminant = ti.sqrt(discriminant)
        t1 = (-B - sqrt_discriminant) / (2 * A)
        t2 = (-B + sqrt_discriminant) / (2 * A)

        if t1 < 0 and t2 < 0:
            has_intersection = False
        else:
            t_intersect = t1 if t1 >= 0 else t2
            if abs(t1 - t2) < eps:
                t_intersect = ti.min(t1, t2)

            intersection_point_transformed = o_transformed + t_intersect * d_transformed
            transform_mat = R @ tm.mat3([
                [S[0], 0, 0],
                [0, S[1], 0],
                [0, 0, S[2]]])
            intersection_point = transform_mat @ intersection_point_transformed + t

            has_intersection = True
    return has_intersection, intersection_point


@ti.func
def get_point_to_line_vector(
    point: tm.vec3,
    line_origin: tm.vec3,
    line_direction: tm.vec3
):
    """ given a point and a line, return the vector from the point to the line

    Args:
        point (tm.vec3): the point, x, y, z
        line_origin (tm.vec3): the origin of the line(ray) in the same space as the point, x, y, z
        line_direction (tm.vec3): the direction of the line(ray), x, y, z

    Returns:
        tm.vec3: the vector from the point to the line
    """
    p = point
    o = line_origin
    d = line_direction
    op = p - o
    scale_factor = ti.math.dot(op, d) / ti.math.dot(d, d)
    q = o + scale_factor * d
    qp = p - q
    return qp
