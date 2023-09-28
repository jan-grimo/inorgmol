use crate::permutation::Permutation;

extern crate nalgebra as na;
use na::base::dimension::{U1, U3};
use nalgebra_lapack::SymmetricEigen;

/// Moments of inertia Ia, Ib, Ic in ascending order
pub struct Moments(pub na::Vector3<f64>);
/// Axes of inertia a, b, c corresponding to [Moments]
pub struct Axes(pub na::Matrix3<f64>);

impl Moments {
    /// Calculate the number of degenerate moments of inertia
    pub fn degeneracy(&self) -> usize {
        const EPSILON: f64 = 0.05;
        let m = self.0;

        let a = if ((m[2] - m[1]) / m[2]).abs() <= EPSILON { 1 } else { 0 };
        let b = if ((m[1] - m[0]) / m[1]).abs() <= EPSILON { 2 } else { 0 };

        1 + (a + b + 1) / 2
    }
}

/// Moments and axes of inertia for a collection of particles
///
/// Does not remove the centroid for these particles. Rotation is about the origin.
pub fn moments_axes(particles: &na::Matrix3xX<f64>) -> (Moments, Axes) {
    let mut inertial_mat = na::Matrix3::<f64>::zeros();

    for col in particles.column_iter() {
        inertial_mat[(0, 0)] += col.y.powi(2) + col.z.powi(2);
        inertial_mat[(1, 1)] += col.x.powi(2) + col.z.powi(2);
        inertial_mat[(2, 2)] += col.x.powi(2) + col.y.powi(2);

        inertial_mat[(1, 0)] -= col.x * col.y;
        inertial_mat[(2, 0)] -= col.x * col.z;
        inertial_mat[(2, 1)] -= col.y * col.z;
    }

    // TODO try removing nalgebra-lapack once nalgebra/{#379, #693 and #1109} are fixed
    // There must be some catastrophic cancellation here in some cases
    let decomposition = SymmetricEigen::new(inertial_mat);
    
    let ordering = Permutation::ordering_by(decomposition.eigenvalues.as_slice(), |a, b| a.partial_cmp(b).expect("No NaNs"));
    if ordering.is_identity() {
        return (Moments(decomposition.eigenvalues), Axes(decomposition.eigenvectors));
    }

    let ordered_eigenvalues = na::Vector3::from_vec(ordering.apply_slice(decomposition.eigenvalues.as_slice()).expect("Matching permutation length"));

    let ordered_eigenvectors = {
        let columns: Vec<_> = decomposition.eigenvectors.column_iter().collect();
        let permuted = ordering.apply_slice(&columns).expect("Matching permutation length");
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
    z: na::Unit<na::Vector3<f64>>,
}

impl CoordinateSystem {
    /// Right-handed constructor from two perpendicular vectors that will become x and y axes
    fn new_xy<S1, S2>(a: Vec3StoredBy<S1>, b: Vec3StoredBy<S2>) -> CoordinateSystem 
        where S1: na::Storage<f64, U3, U1>, 
            S2: na::Storage<f64, U3, U1>
    {
        assert!((angle(&a, &b) - std::f64::consts::FRAC_PI_2).abs() < 1e-4);

        let z = na::Unit::new_normalize(a.cross(&b));

        CoordinateSystem {
            x: na::Unit::new_normalize(a.into_owned()),
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

fn align_to_z(particles: na::Matrix3xX<f64>) -> (Top, na::Matrix3xX<f64>) {
    let (moments, axes) = moments_axes(&particles);
    let degeneracy = moments.degeneracy();

    if degeneracy == 1 {
        /* Asymmetric. Rotate the axis with the highest moment of inertia to
         * coincide with z, and the one with second most to coincide with x.
         */
        let coord = CoordinateSystem::new_xy(
            axes.0.column(1),
            axes.0.column(0)
        );
        let rotated = coord.quaternion().to_rotation_matrix() * particles;
        return (Top::Asymmetric, rotated);
    }

    if degeneracy == 2 {
        // Line: Ia << Ib = Ic and Ia ≃ 0
        if moments.0.x < 0.1 {
            let coord = CoordinateSystem::new_xy(axes.0.column(1), axes.0.column(2));
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
            let coord = CoordinateSystem::new_xy(axes.0.column(1), axes.0.column(2));
            let rotated = coord.quaternion().to_rotation_matrix() * particles;
            return (Top::Prolate, rotated);
        } 

        // Oblate: Ic is unique
        let coord = CoordinateSystem::new_xy(axes.0.column(0), axes.0.column(1));
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

/// Determine top type of particle collection and reorient main axis along +z
///
/// Handles tops differently:
/// - Asymmetric: Axis with largest moment of inertia to z, second most to x
/// - Line: Orient along z
/// - Oblate (disc) and prolate (football): Unique axis along z
/// - Spherical: Arbitrary particle to +z
///
pub fn standardize_top(particles: na::Matrix3xX<f64>) -> (Top, na::Matrix3xX<f64>) {
    let (top, mut reoriented) = align_to_z(particles);

    // There can be asymmetrically distributed particles along z
    if reoriented.row(2).mean() > 0.1 {
        reoriented.row_mut(1).neg_mut();
        reoriented.row_mut(2).neg_mut();
    }

    (top, reoriented)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use crate::inertia::*;
    use crate::shapes::{Name, shape_from_name, SHAPES};
    use crate::permutation::Permutatable;
    extern crate nalgebra as na;

    const EPSILON: f64 = 1e-6;

    #[test]
    fn coordsys_special_cases() {
        // z == z'
        let mixed_x = (na::Vector3::<f64>::x() + 0.3 * na::Vector3::<f64>::y()).normalize();
        let mixed_y = na::Rotation::<f64, 3>::from_axis_angle(&na::Vector3::z_axis(), std::f64::consts::FRAC_PI_2) * mixed_x;
        let along_plus_z = CoordinateSystem::new_xy(mixed_x, mixed_y);
        let rot = along_plus_z.quaternion().to_rotation_matrix();
        approx::assert_relative_eq!(na::Vector3::x(), rot * mixed_x, epsilon=EPSILON);
        approx::assert_relative_eq!(na::Vector3::y(), rot * mixed_y, epsilon=EPSILON);

        // z == -z'
        let along_minus_z = CoordinateSystem::new_xy(mixed_y, mixed_x);
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

            let rot = CoordinateSystem::new_xy(x, y).quaternion().to_rotation_matrix();
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

    #[test]
    fn consistent_z_axis_reorientation() {
        for shape in SHAPES.iter() {
            if shape.name == Name::Line || shape.name == Name::T {
                // Skip line and t-shape, because quaternion fits are degenerate (Cinfv)
                continue;
            }

            let (a_top, a) = standardize_top(shape.coordinates.clone());

            for _ in 0..10 {
                let rot = na::UnitQuaternion::from_axis_angle(
                    &na::Unit::new_normalize(na::Vector3::new_random()),
                    std::f64::consts::TAU * rand::random::<f64>()
                ).to_rotation_matrix();

                let (b_top, b) = standardize_top(rot * shape.coordinates.clone());
                assert_eq!(a_top, b_top);

                // For all rotations of vertex indices, find the quaternion fit 
                // that has the closest fit axis to +z or -z
                let aligned = shape.generate_rotations().into_iter()
                    .any(|rotation| {
                        let map = HashMap::from_iter(rotation.permutation.into_iter());
                        let fit = crate::quaternions::fit_with_map(&a, &b, &map);
                        let angle_near_zero = fit.quaternion.angle().abs() < std::f64::consts::PI / 180.0;
                        // axis() is None if the rotation is zero, which means perfect overlap
                        let axis_near_z = fit.quaternion.axis()
                            .map(|axis| axis.z.abs())
                            .unwrap_or(1.00) > 0.99;

                        angle_near_zero || axis_near_z
                    });

                assert!(aligned, "Shape {} isn't consistently oriented", shape.name);
            }
        }
    }

    #[test]
    fn seesaw() {
        let seesaw = shape_from_name(Name::Seesaw);
        let coords = seesaw.coordinates.clone();
        let (_, a) = standardize_top(coords);
        let z_rot = na::UnitQuaternion::from_axis_angle(
            &na::Vector3::z_axis(),
            std::f64::consts::PI
        ).to_rotation_matrix();
        let rotated = z_rot * a.clone();
        let scramble = Permutation::new_random(rotated.ncols());
        let scrambled = rotated.permute(&scramble).expect("Matching size");
        let (_, b) = standardize_top(scrambled);
        
        println!("{}", crate::shapes::Shape::round_mat(&a));
        println!("{}", crate::shapes::Shape::round_mat(&b));

        // if !b.column_iter().all(|col| {
        //     col.x.abs() < 1e-3 || (col.x.abs() - 0.87) < 0.1
        // }) {
        //     println!("{}", crate::shapes::Shape::round_mat(&standardized));
        // }
    }
}
