use crate::permutation::Permutation;

extern crate nalgebra as na;
use na::base::dimension::{U1, U3};

pub struct Moments(pub na::Vector3<f64>);
pub struct Axes(pub na::Matrix3<f64>);

impl Moments {
    pub fn degeneracy(&self) -> usize {
        const EPSILON: f64 = 0.05;
        let m = self.0;

        let a = if ((m[2] - m[1]) / m[2]).abs() <= EPSILON { 1 } else { 0 };
        let b = if ((m[1] - m[0]) / m[1]).abs() <= EPSILON { 2 } else { 0 };

        1 + (a + b + 1) / 2
    }
}

pub fn moments_axes(particles: &na::Matrix3xX<f64>) -> (Moments, Axes) {
    let mut inertial_mat = na::Matrix3::<f64>::zeros();

    for col in particles.column_iter() {
        inertial_mat[(0, 0)] += col.y.powi(2) + col.z.powi(2);
        inertial_mat[(1, 1)] += col.x.powi(2) + col.z.powi(2);
        inertial_mat[(2, 2)] += col.x.powi(2) + col.y.powi(2);

        let xy = col.x * col.y;
        inertial_mat[(1, 0)] -= xy;
        inertial_mat[(0, 1)] -= xy;

        let xz = col.x * col.z;
        inertial_mat[(2, 0)] -= xz;
        inertial_mat[(0, 2)] -= xz;

        let yz = col.y * col.z;
        inertial_mat[(2, 1)] -= yz;
        inertial_mat[(1, 2)] -= yz;
    }

    let decomposition = na::SymmetricEigen::new(inertial_mat);
    let ordering = Permutation::ordering_by(decomposition.eigenvalues.as_slice(), |a, b| a.partial_cmp(b).expect("No NaNs"));
    if ordering.is_identity() {
        return (Moments(decomposition.eigenvalues), Axes(decomposition.eigenvectors));
    }

    let ordered_eigenvalues = na::Vector3::from_vec(ordering.apply_collect(decomposition.eigenvalues.as_slice()).expect("Matching permutation length"));

    let ordered_eigenvectors = {
        let columns: Vec<_> = decomposition.eigenvectors.column_iter().collect();
        let permuted = ordering.apply(&columns).expect("Matching permutation length");
        na::Matrix3::from_columns(permuted.as_slice())
    };

    (Moments(ordered_eigenvalues), Axes(ordered_eigenvectors))
}

/// Shape of spinning collection of particles
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Top {
    /// Line: 0 ≃ I_a << I_b = I_c
    Linear,
    /// Asymmetric: I_a < I_b < I_c
    Asymmetric,
    /// Prolate (think rugby football): I_a < I_b = I_c
    Prolate,
    /// Oblate (think disc): I_a = I_b < I_c
    Oblate,
    /// Spherical: I_a = I_b = I_c
    Spherical
}

type Vec3StoredBy<S> = na::Matrix<f64, U3, U1, S>;

fn angle<S1, S2>(a: &Vec3StoredBy<S1>, b: &Vec3StoredBy<S2>) -> f64 
    where S1: na::Storage<f64, U3, U1>, 
        S2: na::Storage<f64, U3, U1>
{
    (a.dot(b) / (a.norm() * b.norm())).acos()
}

fn signed_angle(a: &na::Vector3<f64>, b: &na::Vector3<f64>, n: &na::Vector3<f64>) -> f64 {
    angle(a, b).copysign(n.dot(&a.cross(b)))
}

struct CoordinateSystem {
    x: na::Unit<na::Vector3<f64>>,
    // y: na::Unit<na::Vector3<f64>>,
    z: na::Unit<na::Vector3<f64>>,
}

impl CoordinateSystem {
    /// Right-handed constructor from two perpendicular vectors
    fn new<S1, S2>(a: Vec3StoredBy<S1>, b: Vec3StoredBy<S2>) -> CoordinateSystem 
        where S1: na::Storage<f64, U3, U1>, 
            S2: na::Storage<f64, U3, U1>
    {
        assert!((angle(&a, &b) - std::f64::consts::FRAC_PI_2).abs() < 1e-4);

        let z = na::Unit::new_normalize(a.cross(&b));

        CoordinateSystem {
            x: na::Unit::new_normalize(a.into_owned()),
      //      y: na::Unit::new_normalize(b.into_owned()),
            z
        }
    }

    fn quaternion(&self) -> na::UnitQuaternion<f64> {
        self.quaternion_to(&CoordinateSystem::default())
    }

    fn quaternion_to(&self, target: &CoordinateSystem) -> na::UnitQuaternion<f64> {
        // Have to watch out for cases where z.cross(z') is a null vector
        if approx::relative_eq!(self.z, target.z, epsilon = 1e-8) {
            return na::UnitQuaternion::from_axis_angle(
                &self.z,
                signed_angle(&self.x, &target.x, &self.z)
            );
        }

        if approx::relative_eq!(self.z, -target.z, epsilon = 1e-8) {
            let z_angle = -signed_angle(&self.x, &target.x, &self.z);
            return na::UnitQuaternion::from_axis_angle(&self.z, z_angle)
                * na::UnitQuaternion::from_axis_angle(&target.x, std::f64::consts::PI);
        }

        let n = na::Unit::new_normalize(self.z.cross(&target.z));

        let alpha = signed_angle(&self.x, &n, &self.z);
        let beta = signed_angle(&self.z, &target.z, &n);
        let gamma = signed_angle(&n, &target.x, &target.z);

        na::UnitQuaternion::from_axis_angle(&target.z, gamma)
            * na::UnitQuaternion::from_axis_angle(&n, beta)
            * na::UnitQuaternion::from_axis_angle(&self.z, alpha)
    }
}

impl Default for CoordinateSystem {
    fn default() -> CoordinateSystem {
        CoordinateSystem {
            x: na::Unit::new_unchecked(*na::Vector3::x_axis()),
            // y: na::Unit::new_unchecked(*na::Vector3::y_axis()),
            z: na::Unit::new_unchecked(*na::Vector3::z_axis()),
        }
    }
}

/// Determine top type of particle collection and reorient main axis along +z
///
/// Handles tops differently:
/// - Asymmetric: Axis with largest moment of inertia to z, second most to x
/// - Line: Orient along z
/// - Oblate (disc) and prolate (football): Unique axis along z
/// - Spherical: Arbitrary particle to +z
///
pub fn standardize_top(particles: na::Matrix3xX<f64>) -> (Top, na::Matrix3xX<f64>) {
    let (moments, axes) = moments_axes(&particles);
    let degeneracy = moments.degeneracy();

    if degeneracy == 1 {
        /* Asymmetric. Rotate the axis with the highest moment of inertia to
         * coincide with z, and the one with second most to coincide with x.
         */
        let coord = CoordinateSystem::new(
            axes.0.column(1),
            axes.0.column(2).cross(&axes.0.column(1))
        );
        let rotated = coord.quaternion().to_rotation_matrix() * particles;
        return (Top::Asymmetric, rotated);
    }

    if degeneracy == 2 {
        // Line: Ia << Ib = Ic and Ia ≃ 0
        if moments.0.x < 0.1 {
            let coord = CoordinateSystem::new(axes.0.column(1), axes.0.column(2));
            let rotated = coord.quaternion().to_rotation_matrix() * particles;
            return (Top::Linear, rotated);
        }

        // Remaining subsets:
        // - Oblate (disc): Ia = Ib < Ic
        // - Prolate (rugby football): Ia < Ib = Ic
        //
        // Rotate the unique axis to coincide with z (it's probably the site of
        // the highest-order rotation), and one of the degenerate axes to
        // coincide with x.

        let kappa = {
            let a = 1.0 / moments.0.x;
            let b = 1.0 / moments.0.y;
            let c = 1.0 / moments.0.z;
            (2.0 * b - a - c) / (a - c)
        };
        debug_assert!((-1.0..=1.0).contains(&kappa));

        if kappa < 0.0 {
            // Prolate: Ia is unique
            let coord = CoordinateSystem::new(axes.0.column(1), axes.0.column(2));
            let rotated = coord.quaternion().to_rotation_matrix() * particles;
            return (Top::Prolate, rotated);
        }

        // Oblate: Ic is unique
        let coord = CoordinateSystem::new(axes.0.column(0), axes.0.column(1));
        let rotated = coord.quaternion().to_rotation_matrix() * particles;
        return (Top::Oblate, rotated);
    }

    debug_assert!(degeneracy == 3);
    // The top is spherical. Rotate an arbitrary position to +z
    let selected_particle_index = particles.column_iter()
        .enumerate()
        .find_map(|(idx, v)| {
            // Points close to the origin aren't good reference for rotations
            if v.norm_squared() < 1e-2 {
                return None;
            }

            // Can't find a unique rotation for points along minus z
            let along_minus_z = v.z < 0.0 && crate::geometry::axis_distance(&na::Vector3::z_axis(), &v) < 1e-8;
            if along_minus_z {
                return None;
            }

            Some(idx)
        })
        .expect("Found a suitable particle to rotate to +z for spherical top");

    let rotation = na::Rotation::<f64, 3>::rotation_between(
        &particles.column(selected_particle_index),
        &na::Vector3::z_axis()
    ).expect("Can generate rotation for selected particle standardizing spherical top");
    (Top::Spherical, rotation * particles)
}

#[cfg(test)]
mod tests {
    use crate::inertia::*;
    use crate::shapes::{Name, shape_from_name};
    extern crate nalgebra as na;

    const EPSILON: f64 = 1e-6;

    #[test]
    fn coordsys_special_cases() {
        // z == z'
        let mixed_x = (na::Vector3::<f64>::x() + 0.3 * na::Vector3::<f64>::y()).normalize();
        let mixed_y = na::Rotation::<f64, 3>::from_axis_angle(&na::Vector3::z_axis(), std::f64::consts::FRAC_PI_2) * mixed_x;
        let along_plus_z = CoordinateSystem::new(mixed_x, mixed_y);
        let rot = along_plus_z.quaternion().to_rotation_matrix();
        approx::assert_relative_eq!(na::Vector3::x(), rot * mixed_x, epsilon=EPSILON);
        approx::assert_relative_eq!(na::Vector3::y(), rot * mixed_y, epsilon=EPSILON);

        // z == -z'
        let along_minus_z = CoordinateSystem::new(mixed_y, mixed_x);
        let rot = along_minus_z.quaternion().to_rotation_matrix();
        approx::assert_relative_eq!(na::Vector3::x(), rot * mixed_y, epsilon=EPSILON);
        approx::assert_relative_eq!(na::Vector3::y(), rot * mixed_x, epsilon=EPSILON);
    }

    #[test]
    fn coordsys_general_cases() {
        for _ in 0..10 {
            let x = na::Vector3::<f64>::new_random().normalize();
            let v = na::Vector3::<f64>::new_random();
            let y = x.cross(&v).normalize();

            let rot = CoordinateSystem::new(x, y).quaternion().to_rotation_matrix();
            approx::assert_relative_eq!(na::Vector3::x(), rot * x, epsilon = EPSILON);
            approx::assert_relative_eq!(na::Vector3::y(), rot * y, epsilon = EPSILON);
        }
    }

    #[test]
    fn top_identification() {
        let pairs = vec![
            (Name::Line, Top::Linear),
            (Name::Bent, Top::Asymmetric),
            (Name::Seesaw, Top::Asymmetric),
            (Name::TrigonalBipyramid, Top::Prolate),
            (Name::PentagonalBipyramid, Top::Oblate),
            (Name::Square, Top::Oblate),
            (Name::Octahedron, Top::Spherical),
            (Name::Tetrahedron, Top::Spherical),
        ];

        for (name, expected_top) in pairs {
            let shape = shape_from_name(name);
            let (top, _) = standardize_top(shape.coordinates.clone());
            assert_eq!(top, expected_top);
        }
    }

    fn indices_along_z(mat: &na::Matrix3xX::<f64>) -> Vec<usize> {
        mat.column_iter()
            .enumerate()
            .filter_map(|(idx, v)| {
                let z_dist = crate::geometry::axis_distance(&na::Vector3::z_axis(), &v);
                if z_dist < 1e-6 {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect()
    }

    fn test_indices_along_z(shape_name: Name, expected_count: usize) {
        let linear = shape_from_name(shape_name).coordinates.clone();
        let (expected_top, expected_reorientation) = standardize_top(linear.clone());
        let expected_indices_along_z = indices_along_z(&expected_reorientation);
        assert_eq!(expected_indices_along_z.len(), expected_count);

        for _ in 0..3 {
            let random_rotation = na::Rotation3::from_axis_angle(
                &na::Unit::new_normalize(na::Vector3::new_random()),
                rand::random::<f64>() * std::f64::consts::TAU
            );
            let (top, reoriented) = standardize_top(random_rotation * linear.clone());
            assert_eq!(top, expected_top);
            assert_eq!(expected_indices_along_z, indices_along_z(&reoriented));
        }
    }

    #[test]
    fn consistent_z_axis_reorientation() {
        // Tops with unique axes
        test_indices_along_z(Name::Line, 2); // Linear
        test_indices_along_z(Name::TrigonalBipyramid, 2); // Prolate
        test_indices_along_z(Name::PentagonalBipyramid, 2); // Oblate
        test_indices_along_z(Name::Square, 0); // Oblate

        // Tops without unique axes
        // TODO asymmetric and spherical
    }
}
