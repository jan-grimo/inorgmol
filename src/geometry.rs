use rand::Rng;
use itertools::Itertools;

extern crate nalgebra as na;
type Matrix3N = na::Matrix3xX<f64>;
type Vector3 = na::Vector3<f64>;

/// Three-dimensional plane
pub struct Plane {
    /// Normal vector
    pub normal: Vector3,
    /// Offset vector
    pub offset: Vector3
}

impl Plane {
    /// Find the plane of best fit to a cloud of particles
    pub fn fit_matrix(mut cloud: Matrix3N) -> Plane {
        // Remove centroid
        let centroid: Vector3 = cloud.column_sum() / (cloud.ncols() as f64);
        for mut col in cloud.column_iter_mut() {
            col -= centroid;
        }

        // Normal is left singular vector of least singular value
        let u = cloud.svd(true, true).u.expect("SVD worked and computed u");
        // NOTE: guaranteed sorted descending, take last column of u
        let normal: Vector3 = u.column(u.ncols() - 1).into();
        assert!((normal.norm() - 1.0).abs() < 1e-6);
        Plane {normal, offset: centroid}
    }

    /// Find the plane of best fit to a subset of particles in a matrix
    pub fn fit_matrix_points<T: Copy + Into<usize>>(matrix: &Matrix3N, particle_indices: &[T]) -> Plane {
        let n = particle_indices.len();
        let mut plane_vertices = Matrix3N::zeros(n);
        for (i, &v) in particle_indices.iter().enumerate() {
            plane_vertices.set_column(i, &matrix.column(v.into()));
        }

        Self::fit_matrix(plane_vertices)
    }

    /// Signed distance of a point to the plane
    ///
    /// Points in the halfspace indicated by the plane normal have positive distance.
    pub fn signed_distance<S>(&self, point: &na::Matrix<f64, na::Const<3>, na::Const<1>, S>) -> f64 where S: na::Storage<f64, na::Const<3>, na::Const<1>> {
        self.normal.dot(&(point - self.offset))
    }

    /// Root-mean-square deviation of plane fit
    pub fn rmsd(&self, cloud: &Matrix3N, vertices: &[usize]) -> f64 {
        let sum_of_squares: f64 = vertices.iter()
            .map(|&v| self.signed_distance(&cloud.column(v)).powi(2))
            .sum();
        (sum_of_squares / (vertices.len() as f64)).sqrt()
    }

    /// Check whether this plane is parallel to another plane
    pub fn is_parallel_to(&self, other: &Plane) -> bool {
        self.normal.cross(&other.normal).norm() < 1e-6 
            || self.normal.cross(&-other.normal).norm() < 1e-6
    }
}


/// Find perpendicular vector to a point from an axis
pub fn axis_perpendicular_component<S1, S2>(
    axis: &na::Matrix<f64, na::Const<3>, na::Const<1>, S1>, 
    point: &na::Matrix<f64, na::Const<3>, na::Const<1>, S2>) -> Vector3
where S1: na::Storage<f64, na::Const<3>, na::Const<1>>, 
    S2: na::Storage<f64, na::Const<3>, na::Const<1>> {
    point - (point.dot(axis)) * axis
}

/// Find distance of a point from an axis
pub fn axis_distance<S1, S2>(
    axis: &na::Matrix<f64, na::Const<3>, na::Const<1>, S1>, 
    point: &na::Matrix<f64, na::Const<3>, na::Const<1>, S2>) -> f64 
where S1: na::Storage<f64, na::Const<3>, na::Const<1>>, 
    S2: na::Storage<f64, na::Const<3>, na::Const<1>> {
    axis_perpendicular_component(axis, point).norm()
}

/// Calculate the signed tetrahedron volume spanned by four points
pub fn signed_tetrahedron_volume<S>(
    a: na::Matrix<f64, na::Const<3>, na::Const<1>, S>,
    b: na::Matrix<f64, na::Const<3>, na::Const<1>, S>,
    c: na::Matrix<f64, na::Const<3>, na::Const<1>, S>,
    d: na::Matrix<f64, na::Const<3>, na::Const<1>, S>
) -> f64 where S: na::Storage<f64, na::Const<3>, na::Const<1>> + Clone {
    (a - d.clone()).dot(&(b - d.clone()).cross(&(c - d))) / 6.0
}

/// Calculate the signed tetrahedron voluem spanned by four points
pub fn signed_tetrahedron_volume_with_array<S>(positions: [na::Matrix<f64, na::Const<3>, na::Const<1>, S>; 4]) -> f64 where S: na::Storage<f64, na::Const<3>, na::Const<1>> + Clone {
    let [a, b, c, d] = positions;
    signed_tetrahedron_volume(a, b, c, d)
}

#[derive(Default)]
struct Face {
    mask: u32,
    affine: Vec<f64>
}

fn separating_plane_face_a(diff: &na::Matrix3x4<f64>, normal: &na::Vector3<f64>) -> Face {
    let mut mask: u32 = 0;

    let powers: [u32; 4] = [1, 2, 4, 8];

    let affine = diff.column_iter().zip(powers.iter()).map(|(c, p)| {
        // Check if vertex outside half-space defined by normal
        let affine_coord = c.dot(normal);
        if affine_coord > 0.0 {
            mask |= p;
        }
        affine_coord
    }).collect();

    Face {mask, affine}
}

/// Checks if there is a separating plane containing an edge shared by two
/// faces of the tetrahedron a
fn separating_plane_edge_a(a: &Face, b: &Face) -> bool {
    if a.mask | b.mask != 15 {
        // One bit is zero for both masks
        // <-> A vertex of b is inside quadrant (-, -) (inside both halfspaces)
        // <-> No sep. plane w/ edge shared by these faces
        return false;
    }

    // No vertices are in (-, -) established above, so no edges with one or
    // both vertices in quadrants (+, +) can intersect the quadrant (-, -)
    // -> Do not consider any edges of tetrahedron b with one or both vertices
    //    in (+, +)
    let in_opposite_quadrants = a.mask ^ b.mask;
    let mask_a = a.mask & in_opposite_quadrants;
    let mask_b = b.mask & in_opposite_quadrants;

    let try_edge = |i: usize, j: usize| {
        let vp = a.affine[j] * b.affine[i] - a.affine[i] * b.affine[j];
        let m_i = 2_u32.pow(i as u32);
        let m_j = 2_u32.pow(j as u32);

        // Test intersections with quadrant (-, -) by rotation orientation
        ((mask_a & m_i) > 0 && (mask_b & m_j) > 0 && vp > 0.0)
            || ((mask_a & m_j) > 0 && (mask_b & m_i) > 0 && vp < 0.0)
    };

    !(
        try_edge(0, 1) || try_edge(0, 2) || try_edge(0, 3) 
        || try_edge(1, 2) || try_edge(1, 3) 
        || try_edge(2, 3)
    )
}

fn separating_plane_face_b(diff: &na::Matrix3x4<f64>, normal: &na::Vector3<f64>) -> bool {
    diff.column_iter().all(|c| c.dot(normal) > 0.0)
}

/// Tetrahedron-tetrahedron overlap test
///
/// Adapted from "Fast tetrahedron-tetrahedron overlap algorithm" by Fabio Ganovelli,
/// Federico Ponchio and Claudio Rocchini: Fast Tetrahedron-Tetrahedron Overlap Algorithm, Journal
/// of Graphics Tools, 7(2), 2002. DOI: 10.1080/10867651.2002.10487557. Original source code
/// archived at [http://web.archive.org/web/20031130075955/http://www.acm.org/jgt/papers/GanovelliPonchioRocchini02/tet_a_tet.html](web.archive.org)
///
/// NOTE: Finicky edge case: If the two tetrahedra have any matching vertices, this method says
/// they overlap
pub fn tetrahedra_overlap(a: &na::Matrix3x4<f64>, b: &na::Matrix3x4<f64>) -> bool {
    const EPSILON: f64 = 1e-6;
    // Check if there are any identical points (edge case, but call overlap)
    if a.column_iter().cartesian_product(b.column_iter()).any(|(r, s)| (r-s).norm_squared() < EPSILON) {
        return true;
    }

    const ALL_OUTSIDE_HALFSPACE: u32 = 15;

    let mut a_edges = Matrix3N::zeros(5);
    let mut faces: [Face; 4] = Default::default();

    // Face 1
    let diff = { // b - a.col(0)
        let mut tmp = *b;
        tmp.column_iter_mut().for_each(|mut c| {c -= a.column(0);});
        tmp
    };
    a_edges.set_column(0, &(a.column(1) - a.column(0)));
    a_edges.set_column(1, &(a.column(2) - a.column(0)));
    a_edges.set_column(2, &(a.column(3) - a.column(0)));
    let mut normal = a_edges.column(1).cross(&a_edges.column(0));
    if normal.dot(&a_edges.column(2)) > 0.0 {
        normal = -normal;
    }
    faces[0] = separating_plane_face_a(&diff, &normal);
    if faces[0].mask == ALL_OUTSIDE_HALFSPACE {
        return false;
    }

    // Face 2
    normal = a_edges.column(0).cross(&a_edges.column(2));
    if normal.dot(&a_edges.column(1)) > 0.0 {
        normal = -normal;
    }
    faces[1] = separating_plane_face_a(&diff, &normal);
    if faces[1].mask == ALL_OUTSIDE_HALFSPACE || separating_plane_edge_a(&faces[0], &faces[1]) {
        return false;
    }

    // Face 3
    normal = a_edges.column(2).cross(&a_edges.column(1));
    if normal.dot(&a_edges.column(0)) > 0.0 {
        normal = -normal;
    }
    faces[2] = separating_plane_face_a(&diff, &normal);
    if faces[2].mask == ALL_OUTSIDE_HALFSPACE 
        || separating_plane_edge_a(&faces[0], &faces[2]) 
        || separating_plane_edge_a(&faces[1], &faces[2]) 
    { 
        return false; 
    }

    // Face 4
    a_edges.set_column(3, &(a.column(2) - a.column(1)));
    a_edges.set_column(4, &(a.column(3) - a.column(1)));
    // ref: FaceA_2 uses (b - a[1]) instead of (b - a[0])
    let different_diff = {
        let mut tmp = *b;
        tmp.column_iter_mut().for_each(|mut c| c -= a.column(1));
        tmp
    };
    normal = a_edges.column(3).cross(&a_edges.column(4));
    if normal.dot(&a_edges.column(0)) < 0.0 {
        normal = -normal;
    }
    faces[3] = separating_plane_face_a(&different_diff, &normal);
    if faces[3].mask == ALL_OUTSIDE_HALFSPACE 
        || separating_plane_edge_a(&faces[0], &faces[3])
        || separating_plane_edge_a(&faces[1], &faces[3])
        || separating_plane_edge_a(&faces[2], &faces[3]) 
    { 
        if separating_plane_edge_a(&faces[0], &faces[3]) {
        }
        return false; 
    }

    // NOTE: Omissible without impact on correctness
    if faces[0].mask | faces[1].mask | faces[2].mask | faces[3].mask != 15 {
        // At least one bit is zero in all masks 
        // <-> At least one vertex is inside all halfspaces of face planes
        // <-> At least one vertex is inside the tetrahedron
        // <-> The tetrahedra overlap
        return true;
    }


    // So now if there is a separating plane it is parallel to a face of b
    // Face 1
    let diff = { // a - b.col(0)
        let mut tmp = *a;
        tmp.column_iter_mut().for_each(|mut c| { c -= b.column(0); });
        tmp
    };
    let mut b_edges = Matrix3N::zeros(5);
    b_edges.set_column(0, &(b.column(1) - b.column(0)));
    b_edges.set_column(1, &(b.column(2) - b.column(0)));
    b_edges.set_column(2, &(b.column(3) - b.column(0)));
    normal = b_edges.column(1).cross(&b_edges.column(0));
    if normal.dot(&b_edges.column(2)) > 0.0 {
        normal = -normal;
    }
    if separating_plane_face_b(&diff, &normal) {
        return false;
    }

    // Face 2
    normal = b_edges.column(0).cross(&b_edges.column(2));
    if normal.dot(&b_edges.column(1)) > 0.0 {
        normal = -normal;
    }
    if separating_plane_face_b(&diff, &normal) {
        return false;
    }
    
    // Face 3
    normal = b_edges.column(2).cross(&(b_edges.column(1)));
    if normal.dot(&b_edges.column(0)) > 0.0 {
        normal = -normal;
    }
    if separating_plane_face_b(&diff, &normal) {
        return false;
    }

    // Face 4
    b_edges.set_column(3, &(b.column(2) - b.column(1)));
    b_edges.set_column(4, &(b.column(3) - b.column(1)));
    // ref: FaceB_2 uses a - b.col(1)
    let different_diff = {
        let mut tmp = *a;
        tmp.column_iter_mut().for_each(|mut c| c -= b.column(1));
        tmp
    };
    normal = b_edges.column(3).cross(&b_edges.column(4));
    if normal.dot(&b_edges.column(0)) < 0.0 {
        normal = -normal;
    }
    if separating_plane_face_b(&different_diff, &normal) {
        return false;
    }
    
    true
}

/// Relation between a point and tetrahedron in space
#[derive(PartialEq, Debug)]
pub enum PointTetrahedronRelation {
    /// Point is inside tetrahedron
    Inside,
    /// Point is on a face of the tetrahedron
    OnFace,
    /// Point is on an edge of the tetrahedron
    OnEdge,
    /// Point is a vertex of the tetrahedron
    Vertex,
    /// Point is outside the tetrahedron
    Outside
}

/// Determine the geometric relation between a point and a tetrahedron
pub fn point_tetrahedron_relation(tetrahedron: &na::Matrix3x4<f64>, point: &na::Vector3<f64>) -> Option<PointTetrahedronRelation> {
    const EPSILON: f64 = 1e-6;
    let mut a: na::Matrix3<f64> = tetrahedron.fixed_view::<3, 3>(0, 0).into();
    a.column_iter_mut().for_each(|mut c| c -= tetrahedron.column(3));
    let b = point - tetrahedron.column(3);

    if let Some(u) = a.lu().solve(&b) {
        let u4 = 1.0 - u.row_iter().fold(0.0, |acc, v| acc + v[0]);
        let u = u.insert_row(3, u4);

        if u.iter().any(|&v| v < -EPSILON) {
            return Some(PointTetrahedronRelation::Outside);
        }

        let zero_count = u.iter().filter(|&v| v.abs() <= EPSILON).count();

        return match zero_count {
            0 => Some(PointTetrahedronRelation::Inside),
            1 => Some(PointTetrahedronRelation::OnFace),
            2 => Some(PointTetrahedronRelation::OnEdge),
            3 => Some(PointTetrahedronRelation::Vertex),
            _ => panic!("Four items in v cannot be zero simultaneously, one is 1.0 - sum(others)")
        };
    }

    None
}

/// Select a random point in a tetrahedron
///
/// Algorithm from
/// [https://vcg.isti.cnr.it/activities/OLD/geometryegraphics/pointintetraedro.html](vcg.isti.cnr.it)
pub fn random_point_in_tetrahedron(tetrahedron: &na::Matrix3x4<f64>) -> Vector3 {
    let [mut s, mut t, mut u]: [f64; 3] = rand::random();

    // cube into prism
    if s + t > 1.0 {
        s = 1.0 - s;
        t = 1.0 - t;
    }

    // prism into tetrahedron
    if t + u > 1.0 {
        let tmp = u;
        u = 1.0 - s - t;
        t = 1.0 - tmp;
    } else if s + t + u > 1.0 {
        let tmp = u;
        u = s + t + u - 1.0;
        s = 1.0 - t - tmp;
    }

    // a, s, t and u are the barycentric coordinate coefficients
    let a = 1.0 - s - t - u;
    [a, s, t, u].into_iter().zip(tetrahedron.column_iter()).fold(Vector3::zeros(), |acc, (l, c)| acc + l * c)
}

/// Find the approximate overlap volume between two tetrahedra
///
/// Calculates overlap by Monte Carlo integration.
///
/// TODO: rewrite to statistical digit convergence criterion instead of number of samples
pub fn approximate_tetrahedron_overlap_volume(a: &na::Matrix3x4<f64>, b: &na::Matrix3x4<f64>) -> f64 {
    // TODO try std::simd?

    const NUM_SAMPLES: usize = 2000;

    // Figure out bounding box, then monte carlo sample points within the volume
    let bounding_min = Vector3::from_fn(|dim, _| a.row(dim).min().min(b.row(dim).min()));
    let bounding_max = Vector3::from_fn(|dim, _| a.row(dim).max().max(b.row(dim).max()));
    let box_widths = bounding_max - bounding_min;
    debug_assert!(box_widths.iter().all(|&v| v >= 0.0));

    let mut rng = rand::thread_rng();
    let in_overlap_count = (0..NUM_SAMPLES).filter(|_| {
        let sample = {
            let mut tmp = box_widths;
            tmp.iter_mut().for_each(|c| *c *= rng.gen::<f64>());
            bounding_min + tmp
        };
        // Random vector is in bounding box
        debug_assert!(sample.iter().enumerate().all(|(dim, v)| (bounding_min[dim]..=bounding_max[dim]).contains(v)));
        let in_a = point_tetrahedron_relation(a, &sample).expect("Valid tetrahedron a") == PointTetrahedronRelation::Inside;
        let in_b = point_tetrahedron_relation(b, &sample).expect("Valid tetrahedron b") == PointTetrahedronRelation::Inside;
        in_a && in_b
    }).count();

    let bounding_box_volume = box_widths.into_iter().fold(1.0, |acc, v| acc * v);
    debug_assert!(bounding_box_volume > 0.0);
    bounding_box_volume * (in_overlap_count as f64) / (NUM_SAMPLES as f64)
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use crate::shapes::{TETRAHEDRON, SQUARE};
    use crate::geometry::*;
    use crate::permutation::permutations;
    use crate::permutation::Permutatable;

    #[test]
    fn plane() {
        let xy = Plane {
            normal: Vector3::z(),
            offset: Vector3::zeros()
        };
        assert_eq!(xy.signed_distance(&Vector3::z()), 1.0);
        assert_eq!(xy.signed_distance(&-Vector3::z()), -1.0);
        assert_eq!(xy.signed_distance(&(0.5 * Vector3::z())), 0.5);

        let xy = Plane::fit_matrix(SQUARE.coordinates.matrix.clone());
        assert_eq!(xy.signed_distance(&Vector3::z()).abs(), 1.0);
    }

    #[test]
    fn tetrahedron_overlap() {
        // Edge case: These two tetrahedra overlap in one point
        let a = na::Matrix3x4::from_column_slice(&[
             0.000000,  0.000000,  0.000000,
             0.755929,  0.000000,  0.654654,
            -0.377964,  0.654654,  0.654654,
            -0.377964, -0.654654,  0.654654,
        ]);
        let b = na::Matrix3x4::from_column_slice(&[
             0.000000,  0.000000,  0.000000,
             0.755929,  0.000000, -0.654654,
            -0.377964,  0.654654, -0.654654,
            -0.377964, -0.654654, -0.654654
        ]);

        // Ensure symmetry by testing all column permutations of both matrices
        assert!(permutations(4)
            .cartesian_product(permutations(4))
            .all(|(p, q)| {
                let m = a.permute(&p).unwrap();
                let n = b.permute(&q).unwrap();
                tetrahedra_overlap(&m, &n) && tetrahedra_overlap(&n, &m)
            }));


        // A does not overlap with this matrix
        let c = na::Matrix3x4::from_column_slice(&[
             0.000000,  0.000000, -0.100000,
             0.755929,  0.000000, -0.654654,
            -0.377964,  0.654654, -0.654654,
            -0.377964, -0.654654, -0.654654,
        ]);

        assert!(permutations(4)
            .cartesian_product(permutations(4))
            .all(|(p, q)| {
                let m = a.permute(&p).unwrap();
                let n = c.permute(&q).unwrap();
                !tetrahedra_overlap(&m, &n) && !tetrahedra_overlap(&n, &m)
            }));

        // B does overlap with C
        assert!(permutations(4)
            .cartesian_product(permutations(4))
            .all(|(p, q)| {
                let m = b.permute(&p).unwrap();
                let n = c.permute(&q).unwrap();
                tetrahedra_overlap(&m, &n) && tetrahedra_overlap(&n, &m)
            }));

        // Completely enclosing larger tetrahedron
        let tetr: na::Matrix3x4<f64> = TETRAHEDRON.coordinates.matrix.fixed_view::<3, 4>(0, 0).into();
        let tetr_larger = tetr.scale(1.1);
        assert!(tetrahedra_overlap(&tetr_larger, &tetr));
        assert!(tetrahedra_overlap(&tetr, &tetr_larger));
    }

    #[test]
    fn point_tetrahedron() {
        let tetr: na::Matrix3x4<f64> = TETRAHEDRON.coordinates.matrix.fixed_view::<3, 4>(0, 0).into();

        assert_eq!(
            point_tetrahedron_relation(&tetr, &Vector3::zeros()), 
            Some(PointTetrahedronRelation::Inside)
        );

        let z = Vector3::from_column_slice(&[0.0, 0.0, 1.0]);
        assert_eq!(
            point_tetrahedron_relation(&tetr, &z), 
            Some(PointTetrahedronRelation::Vertex)
        );

        let on_edge = tetr.column(0) + 0.6 * (tetr.column(1) - tetr.column(0));
        assert_eq!(
            point_tetrahedron_relation(&tetr, &on_edge), 
            Some(PointTetrahedronRelation::OnEdge)
        );

        let on_face = tetr.column(0) 
            + 0.4 * (tetr.column(1) - tetr.column(0))
            + 0.3 * (tetr.column(2) - tetr.column(0));
        assert_eq!(
            point_tetrahedron_relation(&tetr, &on_face), 
            Some(PointTetrahedronRelation::OnFace)
        );

        let outside = tetr.column(0).scale(1.1);
        assert_eq!(
            point_tetrahedron_relation(&tetr, &outside), 
            Some(PointTetrahedronRelation::Outside)
        );
    }

    #[test]
    fn random_point_in_tetrahedron_joint() {
        // Symmetric tetrahedron
        let tetr: na::Matrix3x4<f64> = TETRAHEDRON.coordinates.matrix.fixed_view::<3, 4>(0, 0).into();
        assert!((0..20).all(|_| {
            point_tetrahedron_relation(&tetr, &random_point_in_tetrahedron(&tetr)).unwrap()
                != PointTetrahedronRelation::Outside
        }));

        // Tetrahedron with two vertices coinciding with x axis (possibly an
        // edge case in decomposition)
        let skew_tetr = na::Matrix3x4::from_column_slice(&[
             1.0,  0.0,  0.0,
            -1.0,  0.0,  0.0,
             0.0,  1.0,  0.0,
             0.0,  0.0,  1.0
        ]);
        assert!((0..20).all(|_| {
            point_tetrahedron_relation(&skew_tetr, &random_point_in_tetrahedron(&skew_tetr)).unwrap()
                != PointTetrahedronRelation::Outside
        }));
    }

    #[test]
    fn tetrahedron_overlap_volume() {
        // NOTE: This might be particularly problematic since two vertices
        // decompose more or less to +x/-x, need to check out nalgebras
        // decomposition methods to ensure I'm not doing something stupid
        let a = na::Matrix3x4::from_column_slice(&[
             1.0,  0.0,  0.0,
            -1.0,  0.0,  0.0,
             0.0,  1.0,  0.0,
             0.0,  0.0,  1.0
        ]);
        // Tetrahedron b completely encloses a
        let b = a.scale(1.5);

        // Overlap volume should equal the volume of a
        let overlap = approximate_tetrahedron_overlap_volume(&a, &b);
        let expected = signed_tetrahedron_volume(a.column(0), a.column(1), a.column(2), a.column(3));
        let relative_error = (overlap - expected) / expected;
        assert!(relative_error <= 0.2); // 20% error is fine

        // Completely enclosed regular tetrahedron
        let a: na::Matrix3x4<f64> = TETRAHEDRON.coordinates.matrix.fixed_view::<3, 4>(0, 0).into();
        let b = a.scale(1.1);
        let overlap = approximate_tetrahedron_overlap_volume(&a, &b);
        let expected = signed_tetrahedron_volume(a.column(0), a.column(1), a.column(2), a.column(3)).abs();
        let relative_error = (overlap - expected) / expected;
        assert!(relative_error <= 0.2); // 20% error is fine
    }
}
