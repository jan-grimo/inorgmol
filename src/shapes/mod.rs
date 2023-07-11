extern crate nalgebra as na;
type Matrix3N = na::Matrix3xX<f64>;
type Vector3 = na::Vector3<f64>;

use std::collections::{HashSet, HashMap};
use itertools::Itertools;
use petgraph::unionfind::UnionFind;
use thiserror::Error;
use ordered_float::NotNan;

use crate::strong::{Index, NewTypeIndex};
use crate::geometry::{Plane, axis_distance, axis_perpendicular_component};
use crate::permutation::{Permutation, permutations};
use crate::shapes::similarity::{unit_sphere_normalize, apply_permutation};

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum Name {
    // 2
    Line,
    Bent,
    // 3
    EquilateralTriangle,
    VacantTetrahedron,
    T,
    // 4
    Tetrahedron,
    Square,
    Seesaw,
    TrigonalPyramid,
    // 5
    SquarePyramid,
    TrigonalBipyramid,
    Pentagon,
    // 6
    Octahedron,
    TrigonalPrism,
    PentagonalPyramid,
    Hexagon,
    // 7
    PentagonalBipyramid,
    CappedOctahedron,
    CappedTrigonalPrism,
    // 8
    SquareAntiprism,
    Cube,
    TrigonalDodecahedron,
    HexagonalBipyramid,
    // 9
    TricappedTrigonalPrism,
    CappedSquareAntiprism,
    HeptagonalBipyramid,
    // 10
    BicappedSquareAntiprism,
    // 11
    EdgeContractedIcosahedron,
    // 12
    Icosahedron,
    Cuboctahedron,
    // 13
    // Thirteen,
    // 14
    // BicappedHexagonalAntiprism,
    // 15
    // Fifteen,
    // 16
    // TriangularFaceSixteen,
    // OpposingSquaresSixteen,
}

impl Name {
    pub fn repr(&self) -> &'static str {
        use Name::*;

        match self {
            Line => "line",
            Bent => "bent",
            EquilateralTriangle => "triangle",
            VacantTetrahedron => "vacant tetrahedron",
            T => "T-shaped",
            Tetrahedron => "tetrahedron",
            Square => "square",
            Seesaw => "seesaw",
            TrigonalPyramid => "trigonal pyramid",
            SquarePyramid => "square pyramid",
            TrigonalBipyramid => "trigonal bipyramid",
            Pentagon => "pentagon",
            Octahedron => "octahedron",
            TrigonalPrism => "trigonal prism",
            PentagonalPyramid => "pentagonal pyramid",
            Hexagon => "hexagon",
            PentagonalBipyramid => "pentagonal bipyramid",
            CappedOctahedron => "capped octahedron",
            CappedTrigonalPrism => "capped trigonal prism",
            SquareAntiprism => "square antiprism",
            Cube => "cube",
            TrigonalDodecahedron => "trigonal dodecahedron",
            HexagonalBipyramid => "hexagonal bipyramid",
            TricappedTrigonalPrism => "tricapped trigonal prism",
            CappedSquareAntiprism => "capped square antiprism",
            HeptagonalBipyramid => "heptagonal bipyramid",
            BicappedSquareAntiprism => "bicapped square antiprism",
            EdgeContractedIcosahedron => "edge contracted icosahedron",
            Icosahedron => "icosahedron",
            Cuboctahedron => "cuboctahedron",
            // Name::Thirteen => "thirteen",
            // Name::BicappedHexagonalAntiprism => "bicapped hexagonal antiprism",
            // Name::Fifteen => "fifteen",
            // Name::TriangularFaceSixteen => "triangular sixteen",
            // Name::OpposingSquaresSixteen => "opposing square sixteen",
        }
    }
}

impl std::fmt::Display for Name {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.repr())
    }
}

use crate::strong::bijection::Bijection;

/// Index wrapper for shape vertices
#[derive(Index, Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Vertex(usize);

impl std::fmt::Display for Vertex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vertex({})", self.get())
    }
}

/// A rotation of vertices is the equivalent of an SO(3) rotation in vertex space
pub type Rotation = Bijection<Vertex, Vertex>;
/// A mirror is a sigma symmetry element of a shape, in vertex space
pub type Mirror = Bijection<Vertex, Vertex>;

/// Shape particles are either vertices, or the implicit origin
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Particle {
    Vertex(Vertex),
    Origin
}

/// Construct a particle from a vertex option. If `None`, yields the implicit origin
impl From<Option<Vertex>> for Particle {
    fn from(maybe_vertex: Option<Vertex>) -> Particle {
        match maybe_vertex {
            Some(v) => Particle::Vertex(v),
            None => Particle::Origin
        }
    }
}

impl std::fmt::Display for Particle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Particle::Vertex(v) => write!(f, "{}", v),
            Particle::Origin => write!(f, "origin")
        }
    }
}

// TODO move to where needed
#[derive(Index, Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Column(usize);

pub type Tetrahedron = [Particle; 4];

/// A coordination polyhedron
pub struct Shape {
    pub name: Name,
    /// Unit sphere coordinates without a centroid
    pub coordinates: Matrix3N,
    /// Spatial rotational basis expressed by vertex permutations
    pub rotation_basis: Vec<Rotation>,
}

/// A plane in three dimensions that contains vertices of a shape
pub struct VertexPlane {
    /// Shape vertices contained in the plane
    pub vertices: Vec<Vertex>,
    /// Three dimensional plane found containing `vertices`
    pub plane: Plane
}

/// Reasons why a shape's coordinates could be invalid
#[derive(Error, Debug, PartialEq, Eq)]
pub enum InvalidVerticesError {
    #[error("Coordinates contain an explicit centroid. Centroids are not vertices and are implicitly at the origin")]
    ExplicitCentroid,
    #[error("Shape vertices are not unit spherical")]
    NotUnitSpherical,
    #[error("There are duplicate vertices in the shape coordinates")]
    DuplicateVertices,
    #[error("Existing routine for canonicalization is insufficient")]
    CannotCanonicalize
}

impl Shape {
    fn is_unit_spherical(coordinates: &Matrix3N) -> bool {
        const MAX_COLUMN_DEVIATION: f64 = 1e-6;
        coordinates.column_iter()
            .all(|col| (col.norm() - 1.0).abs() <= MAX_COLUMN_DEVIATION)
    }

    fn has_duplicate_vertices(coordinates: &Matrix3N) -> bool {
        let n = coordinates.ncols();
        (0..n).tuple_combinations().any(|(i, j)| {
            (coordinates.column(i) - coordinates.column(j)).norm_squared() < 1e-3 
        })
    }

    fn try_canonicalize_rotation(canon: &Matrix3N, rotation: &na::Rotation3::<f64>) -> bool {
        let rotated = rotation * canon;
        let scramble = Permutation::new_random(canon.ncols());
        let scrambled = apply_permutation(&rotated, &scramble);
        let maybe_canon = Self::canonicalize_coordinates(scrambled);

        let msd = canon.column_iter()
            .zip(maybe_canon.column_iter())
            .map(|(r_i, r_j)| (r_i - r_j).norm_squared())
            .sum::<f64>() / canon.ncols() as f64;

        dbg!(msd) < 1e-3
    }

    fn can_canonicalize(coordinates: &Matrix3N) -> bool {
        // Some systematic canonicalization tests
        // - Rotation around z
        // - z axis inversion (by rotation around x)
        //
        // 10 random mixed rotations

        let canon = Self::canonicalize_coordinates(coordinates.clone());

        let z_rot = na::UnitQuaternion::from_axis_angle(
            &na::Vector3::z_axis(),
            std::f64::consts::PI
        ).to_rotation_matrix();

        if !Self::try_canonicalize_rotation(&canon, &z_rot) {
            println!("Failed z rot test");
            return false;
        }

        let inv_z_rot = na::UnitQuaternion::from_axis_angle(
            &na::Vector3::x_axis(),
            std::f64::consts::PI
        ).to_rotation_matrix();

        if !Self::try_canonicalize_rotation(&canon, &inv_z_rot) {
            println!("Failed z inversion test");
            return false;
        }

        // Three mixed rotations
        for _ in 0..3 {
            let rot = na::UnitQuaternion::from_axis_angle(
                &na::Unit::new_normalize(Vector3::new_random()),
                std::f64::consts::TAU * rand::random::<f64>()
            ).to_rotation_matrix();

            if !Self::try_canonicalize_rotation(&canon, &rot) {
                println!("Failed random rotation");
                return false;
            }
        }

        true
    }

    /// Construct a new shape
    ///
    /// Constructs a new shape from a name and coordinates.
    ///
    /// Coordinates must:
    /// - be unit spherical with a low tolerance
    /// - not contain an explicit centroid vertex
    /// - not have duplicate vertices
    pub fn try_new(name: Name, coordinates: Matrix3N) -> Result<Shape, InvalidVerticesError> {
        if coordinates.column_iter().any(|v| v.norm_squared() < 1e-3) {
            return Err(InvalidVerticesError::ExplicitCentroid);
        }

        if !Self::is_unit_spherical(&coordinates) {
            return Err(InvalidVerticesError::NotUnitSpherical);
        }
    
        // Ensure no coordinates are present twice
        if Self::has_duplicate_vertices(&coordinates) {
            return Err(InvalidVerticesError::DuplicateVertices);
        }

        if !Self::can_canonicalize(&coordinates) {
            return Err(InvalidVerticesError::CannotCanonicalize);
        }
        
        // Find coordinates-derived properties
        let rotation_basis = Self::find_rotation_basis(&coordinates);
    
        Ok(Shape {name, coordinates, rotation_basis})
    }

    /// Number of vertices of the shape (not including the implicit origin)
    pub fn num_vertices(&self) -> usize {
        self.coordinates.ncols()
    }

    /// Yields the position of a particle
    pub fn particle_position(&self, particle: Particle) -> Vector3 {
        match particle {
            Particle::Vertex(v) => self.coordinates.column(v.get()).into(),
            Particle::Origin => Vector3::zeros()
        }
    }

    /// Expands a basis of rotations into all possible combinations
    fn expand_rotation_basis(basis: &[Rotation]) -> HashSet<Rotation> {
        let mut rotations: HashSet<Rotation> = HashSet::new();

        if basis.is_empty() {
            return rotations;
        }

        let num_vertices = basis.first().unwrap().set_size();

        rotations.insert(Rotation::identity(num_vertices));
        let max_basis_idx = basis.len() - 1;

        struct Frame {
            permutation: Rotation,
            next_basis: usize
        }

        let mut stack = Vec::<Frame>::new();
        stack.push(Frame {permutation: Rotation::identity(num_vertices), next_basis: 0});

        // Tree-like traversal, while tracking rotations applied to get new rotations and pruning
        // if rotations have been seen before
        while stack.first().unwrap().next_basis <= max_basis_idx {
            let latest = stack.last().unwrap();
            let next_rotation = &basis[latest.next_basis];
            let generated = latest.permutation.compose(next_rotation).unwrap();

            if rotations.insert(generated.clone()) {
                // Continue finding new things from this structure
                stack.push(Frame {permutation: generated, next_basis: 0});
            } else {
                // Try to pop unincrementable stack frames
                while stack.len() > 1 && stack.last().unwrap().next_basis == max_basis_idx {
                    stack.pop();
                }

                stack.last_mut().unwrap().next_basis += 1;
            }
        }

        rotations
    }

    /// Generate a full set of rotations from a shape's rotational basis
    ///
    /// ```
    /// # use molassembler::shapes::*;
    /// # use molassembler::strong::bijection::bijections;
    /// # use std::collections::HashSet;
    /// # use std::iter::FromIterator;
    /// let line_rotations = LINE.generate_rotations();
    /// assert_eq!(line_rotations, HashSet::from_iter(bijections(2)));
    /// assert!(line_rotations.iter().all(|r| r.set_size() == 2));
    ///
    /// let tetrahedron_rotations = TETRAHEDRON.generate_rotations();
    /// assert_eq!(tetrahedron_rotations.len(), 12);
    /// assert!(tetrahedron_rotations.iter().all(|r| r.set_size() == 4));
    ///
    /// ```
    pub fn generate_rotations(&self) -> HashSet<Rotation> {
        Self::expand_rotation_basis(self.rotation_basis.as_slice())
    }

    /// Test whether one vertex bijection is a rotation of another
    pub fn is_rotation<T: NewTypeIndex>(a: &Bijection<Vertex, T>, b: &Bijection<Vertex, T>, rotations: &HashSet<Rotation>) -> bool {
        rotations.iter().any(|r| r.compose(a).expect("Bad occupations") == *b)
    }

    /// Make explicit groups of indices out of a UnionFind instance
    fn union_to_groups(sets: UnionFind<usize>) -> Vec<Vec<Vertex>> {
        let group_injection: Vec<usize> = {
            let injection: Vec<usize> = sets.into_labeling();

            // Normalize in one pass
            let mut label_map = HashMap::new();
            let mut unused_group = 0;
            injection.iter().map(|label| {
                if let Some(target) = label_map.get(label) {
                    *target
                } else {
                    let target = unused_group;
                    label_map.insert(label, unused_group);
                    unused_group += 1;
                    target
                }
            }).collect()
        };

        let num_groups = group_injection.iter().max().expect("Empty injection") + 1;
        let mut groups: Vec<Vec<Vertex>> = Vec::new();
        groups.resize(num_groups, Vec::<Vertex>::new());
        for (v, g) in group_injection.iter().enumerate() {
            groups[*g].push(Vertex::from(v));
        }

        groups
    }

    /// Subsets vertices according to superposability by rotation
    pub fn vertex_groups(&self) -> Vec<Vec<Vertex>> {
        let shape_size = self.num_vertices();
        let mut sets = UnionFind::new(shape_size);
        for rotation in self.rotation_basis.iter() {
            for v in 0..shape_size {
                sets.union(v, rotation.permutation[v]);
            }
        }

        Self::union_to_groups(sets)
    }

    pub fn vertex_groups_holding(&self, held: &[Vertex], rotations: &HashSet<Rotation>) -> Vec<Vec<Vertex>> {
        let shape_size = self.num_vertices();
        let mut sets = UnionFind::new(shape_size);
        for rotation in rotations.iter().filter(|rot| held.iter().all(|v| rot.is_fixed_point(*v))) {
            for v in 0..shape_size {
                sets.union(v, rotation.permutation[v]);
            }
        }

        Self::union_to_groups(sets)
    }

    /// Collect coplanar vertices and their joint plane
    fn coplanar_vertex_groups(coordinates: &Matrix3N) -> HashMap<Vec<Vertex>, Plane> {
        let mut groups = HashMap::new();

        let n = coordinates.ncols();
        for mut vertex_triple in (0..n).map(Vertex::from).combinations(3) {
            let mut plane = Plane::fit_matrix_points(coordinates, &vertex_triple);
            let mut coplanar_vertices: Vec<Vertex> = (0..n).map(Vertex::from)
                .filter(|v| !vertex_triple.contains(v) && plane.signed_distance(&coordinates.column(v.get())).abs() < 1e-3)
                .collect();

            if !coplanar_vertices.is_empty() {
                coplanar_vertices.append(&mut vertex_triple);
                coplanar_vertices.sort();
                let mut updated_centroid = Vector3::zeros();
                for &v in coplanar_vertices.iter() {
                    updated_centroid += coordinates.column(v.into());
                }
                updated_centroid /= coplanar_vertices.len() as f64;
                plane.offset = updated_centroid;

                groups.insert(coplanar_vertices, plane);
            } else {
                groups.insert(vertex_triple, plane);
            }

        }

        groups
    }

    /// Collect groups of vertices that are coplanar
    fn parallel_vertex_planes(coordinates: &Matrix3N) -> Vec<Vec<VertexPlane>> {
        let groups = Self::coplanar_vertex_groups(coordinates);
        // Group the planes by normal vector collinearity (should be coaxial)
        let flat: Vec<VertexPlane> = groups.into_iter().map(|(vertices, plane)| VertexPlane {vertices, plane}).collect();
        let f = flat.len();
        let mut parallel_plane_set_finder: UnionFind<usize> = UnionFind::new(f);
        for (i, a) in flat.iter().enumerate() {
            for (j, b) in flat.iter().enumerate().skip(i + 1) {
                if a.plane.parallel(&b.plane) {
                    parallel_plane_set_finder.union(i, j);
                }
            }
        }

        let parallel_labels = parallel_plane_set_finder.into_labeling();

        let mut planes = Vec::new();
        let mut label_map = HashMap::new();
        let mut unused_group = 0;
        for (vertex_plane, label) in flat.into_iter().zip(parallel_labels.into_iter()) {
            let group = {
                if let Some(target) = label_map.get(&label) {
                    &mut planes[*target]
                } else {
                    let target = unused_group;
                    label_map.insert(label, unused_group);
                    unused_group += 1;
                    planes.push(Vec::new());
                    &mut planes[target]
                }
            };
            group.push(vertex_plane);
        }

        planes
    }

    /// Test whether a permutation is a rotation by quaternion fit
    fn permutation_is_rotation(coordinates: &Matrix3N, permutation: &Permutation) -> bool {
        let normalized_points = unit_sphere_normalize(coordinates.clone());
        let permuted = unit_sphere_normalize(apply_permutation(&normalized_points, permutation));
        let fit = crate::quaternions::fit(&normalized_points, &permuted);
        fit.msd < 1e-6
    }

    /// Finds proper rotations containing at least one coplanar set of at least three
    /// vertices. I.e. can find rotations of order two if it also happens to rotate a coplanar
    /// vertex group of four vertices, but cannot find all rotations of order two generally. Mostly
    /// finds rotations of order three and above.
    fn try_parallel_vertex_plane_axis(coordinates: &Matrix3N, planes: &[VertexPlane]) -> Option<Permutation> {
        let minimal_order = planes.iter()
            .map(|p| p.vertices.len())
            .min()
            .expect("At least one per group");
        let order_compatible = planes.iter()
            .all(|p| p.vertices.len() % minimal_order == 0);

        if !order_compatible {
            return None;
        }

        let axis = planes.first().expect("At least one per group").plane.normal;
        let n = coordinates.ncols();
        let mut remaining_vertices: Vec<Vertex> = (0..n).map_into::<Vertex>()
            .filter(|v| !planes.iter().any(|p| p.vertices.contains(v)))
            .collect();

        // TODO remove axial vertices first? Fails faster

        // If the minimal order is two, look for axis-perpendicular dipoles
        // Those are neither in the plane vertex triples nor are axial
        let mut dipoles: Vec<(Vertex, Vertex)> = Vec::new();
        if minimal_order % 2 == 0 {
            'outer: for (i, &a) in remaining_vertices.iter().enumerate() {
                let component_a = axis_perpendicular_component(&axis, &coordinates.column(a.into()));
                if component_a.norm_squared() < 1e-3 {
                    continue;
                }

                for &b in remaining_vertices.iter().skip(i + 1) {
                    let component_b = axis_perpendicular_component(&axis, &coordinates.column(b.into()));
                    if component_b.norm_squared() < 1e-3 {
                        continue;
                    }

                    // Must be antiparallel!
                    if component_a.cross(&-component_b).norm() < 1e-6 {
                        dipoles.push((a, b));
                        // a can't be a dipole w/ anything else
                        continue 'outer;
                    }
                }
            }
            // Remove found dipoles from remaining_vertices
            remaining_vertices.retain(|v| !dipoles.iter().any(|(a, b)| a == v || b == v));
        }

        // The remaining vertices must be along the axis
        let remainder_axial = remaining_vertices.iter()
            .all(|v| axis_distance(&axis, &coordinates.column(v.get())) < 1e-3);

        if !remainder_axial {
            return None;
        }

        // Try lowest order rotation (proper and improper)
        let mut proper_sigma: Vec<_> = (0..n).collect();
        for p in planes.iter() {
            let anchor = p.vertices.first().expect("Guaranteed three vertices per group");
            let positive_signed_angle = |v: Vertex| {
                let a = coordinates.column(anchor.get());
                let b = coordinates.column(v.get());
                let mut angle = a.cross(&b).dot(&axis).atan2(a.dot(&b));
                if angle < 0.0 {
                    angle += std::f64::consts::TAU;
                }
                NotNan::new(angle).expect("Signed angle isn't NaN")
            };
            let mut with_signed_angles: Vec<_> = p.vertices.iter().skip(1)
                .map(|&v| (v, positive_signed_angle(v))).collect();
            with_signed_angles.sort_by_key(|tup| tup.1);

            let mut ordered_vertices: Vec<Vertex> = with_signed_angles.into_iter()
                .map(|(v, _)| v)
                .collect();
            ordered_vertices.push(*anchor);

            for (&i, &j) in ordered_vertices.iter().circular_tuple_windows() {
                proper_sigma[i.get()] = j.get();
            }
        }
        for (i, j) in dipoles.into_iter() {
            proper_sigma[i.get()] = j.get();
            proper_sigma[j.get()] = i.get();
        }
        let proper = Permutation::try_from(proper_sigma).expect("Valid permutation");

        // TODO maybe unnecessary to test
        if Self::permutation_is_rotation(coordinates, &proper) && proper.index() != 0 {
            Some(proper)
        } else {
            None
        }
    }

    /// Test whether a vector is a suitable c2 axis
    fn try_c2_axis(coordinates: &Matrix3N, axis: Vector3) -> Option<Permutation> {
        let rotated = nalgebra::Rotation3::from_axis_angle(
            &nalgebra::Unit::new_normalize(axis),
            std::f64::consts::PI
        ) * coordinates.clone();

        let num_vertices = coordinates.ncols();
        let mut proper_sigma: Vec<_> = (0..num_vertices).collect();
        'outer: for i in 0..num_vertices {
            if proper_sigma[i] != i {
                continue;
            }

            for j in i..num_vertices {
                if proper_sigma[j] != j {
                    continue;
                }

                let distance = (rotated.column(i) - coordinates.column(j)).norm();
                if distance < 0.1 {
                    proper_sigma[i] = j;
                    proper_sigma[j] = i;

                    continue 'outer;
                }
            }

            return None;
        }

        let proper = Permutation::try_from(proper_sigma).expect("Valid permutation");

        if proper.index() != 0 && Self::permutation_is_rotation(coordinates, &proper) {
            Some(proper)
        } else {
            None
        }
    }

    /// Try to find all c2 axes in a shape
    fn find_c2_axes(coordinates: &Matrix3N) -> Vec<Permutation> {
        let mut rotations: Vec<Permutation> = Vec::new();
        for (i, col_i) in coordinates.column_iter().enumerate() {
            // Try each particle position
            if let Some(rotation) = Self::try_c2_axis(coordinates, col_i.into()) {
                rotations.push(rotation);
            }

            // Try each sum of two particle positions, if they don't cancel
            for col_j in coordinates.column_iter().skip(i + 1) {
                let sum = col_i + col_j;
                // Canceling vectors give wild new normalized points, avoid:
                if sum.norm_squared() > 1e-6 {
                    if let Some(rotation) = Self::try_c2_axis(coordinates, sum.normalize()) {
                        rotations.push(rotation);
                    }
                }
            }
        }

        rotations
    }

    /// Reduce a set of rotations to a basis that generates all
    fn reduce_to_basis(mut rotations: HashSet<Rotation>) -> Vec<Rotation> {
        // Find minimal rotations necessary to generate the most rotations
        let min_fixed_rot = rotations.iter()
            .max_by_key(|r| r.permutation.cycles().iter().map(|c| c.len()).sum::<usize>())
            .unwrap()
            .clone();
        rotations.remove(&min_fixed_rot);
        let mut basis = vec![min_fixed_rot];
        let mut basis_size = Self::expand_rotation_basis(basis.as_slice()).len();

        loop {
            let maybe_next_best = rotations.iter()
                .fold(
                    None,
                    |best, r| {
                        basis.push(r.clone());
                        let new_basis_size = Self::expand_rotation_basis(basis.as_slice()).len();
                        let r_copy = basis.pop().expect("Just pushed, definitely there");
                        let delta = new_basis_size - basis_size;
                        let best_delta = best.as_ref().map_or_else(|| 0, |(_, delta)| *delta);
                        if delta > 0 && delta > best_delta {
                            return Some((r_copy, delta));
                        }

                        best
                    }
                );

            if let Some((best_rotation, delta)) = maybe_next_best {
                rotations.remove(&best_rotation);
                basis.push(best_rotation);
                basis_size += delta;
            } else {
                break;
            }
        }

        basis
    }

    /// Find rotations exhaustively by permutational quaternion fit
    fn stupid_find_rotations(coordinates: &Matrix3N) -> Vec<Rotation> {
        // TODO this is really very stupid, as 
        // - repeated applications of true rotations aren't excluded / skipped
        // - combinations of found rotations aren't excluded / skipped
        permutations(coordinates.ncols())
            .filter(|p| Self::permutation_is_rotation(coordinates, p))
            .map(Rotation::new)
            .collect()
    }

    /// Find rotation basis of a shape
    pub fn find_rotation_basis(coordinates: &Matrix3N) -> Vec<Rotation> {
        let plane_axes: Vec<Permutation> = Self::parallel_vertex_planes(coordinates).into_iter()
            .filter_map(|v| Self::try_parallel_vertex_plane_axis(coordinates, v.as_slice()))
            .collect();
        let c2_axes = Self::find_c2_axes(coordinates);

        // plane axes and c2 axes may have overlapping rotations, avoid duplicates
        let mut rotations = plane_axes.into_iter()
            .chain(c2_axes.into_iter())
            .map(Rotation::new)
            .collect::<HashSet<Rotation>>();

        if rotations.is_empty() {
            if coordinates.ncols() > 4 {
                panic!("No rotations found heuristically, too large to use exhaustive search");
            }

            rotations = HashSet::from_iter(Self::stupid_find_rotations(coordinates).into_iter());
        }

        Self::reduce_to_basis(rotations)
    }

    /// Mirror symmetry element expressed by vertex permutation, if present
    pub fn find_mirror(&self) -> Option<Mirror> {
        // TODO This is a bit weird, actually a full inversion, maybe
        // better to just invert a row, e.g. y coordinates?
        let inverted = -1.0 * self.coordinates.clone();
        let inverted = inverted.insert_column(self.num_vertices(), 0.0);

        // if the quaternion fit onto the original coordinates isn't zero-cost,
        // there is a mirror element: the permutation from similarity-fitting
        // the inverted coordinates onto the original ones
        //
        // TODO Is there a better (faster) way?
        let similarity = similarity::polyhedron(inverted, self).ok()?;
        if similarity.csm < 1e-6 && !similarity.bijection.permutation.is_identity() {
            let mut permutation = similarity.bijection.permutation;
            assert!(permutation.pop_if_fixed_point() == Some(self.num_vertices()));
            Some(Mirror::new(permutation))
        } else {
            None
        }
    }

    /// Minimal set of tetrahedra required to distinguish volumes in DG
    pub fn find_tetrahedra(&self) -> Vec<Tetrahedron> {
        // Requirements:
        // - Need to have significant (positive, for norming) volumes to be
        //   efficient in DG.
        // - All vertices involved in non-zero volume quadruples need to be 
        //   covered, but not the origin
        //
        // Idea: Find smallest set of positive volume tetrahedra covering 
        // all vertices (not the origin) minimizing the overlap overlap between
        // tetrahedra vertices
        //
        // Notes:
        // - Regularity/symmetry of tetrahedra would be great
        // - Reducing volume overlap would be good, too, e.g.
        //   2x origin-coplanar triple quads in trig. prism instead of two
        //   overlapping tetrahedra between coplanar sets

        const MIN_TETRAHEDRON_VOLUME: f64 = 0.4 / 6.0;

        let zero = Vector3::zeros();
        let particle_position = |p: &Particle| {
            match p {
                Particle::Vertex(v) => self.coordinates.column(v.get()),
                Particle::Origin => zero.column(0)
            }
        };

        let n = self.num_vertices();
        let mut particles: Vec<Particle> = (0..n).map_into::<Vertex>().map(Particle::Vertex).collect();
        particles.push(Particle::Origin);

        // Find non-zero volume particle quads
        type Quad<'a> = [&'a Particle; 4];
        let mut quads: Vec<Quad> = Vec::new();
        for particle_vec in particles.iter().combinations(4) {
            let mut particle_quad: Quad = particle_vec.try_into().expect("Matched vec size");
            let volume = crate::geometry::signed_tetrahedron_volume_with_array(particle_quad.map(particle_position));
            if volume.abs() > MIN_TETRAHEDRON_VOLUME {
                // Ensure positive volume
                if volume < 0.0 {
                    particle_quad.as_mut_slice().swap(2, 3);
                }

                quads.push(particle_quad);
            }
        }

        if quads.is_empty() {
            return Vec::new();
        }

        let quad_vertex_overlap = |a: Quad, b: Quad| -> usize {
            let mut particles = Vec::from_iter(a.iter().chain(b.iter()));
            particles.sort();
            particles.iter().tuple_windows().filter(|(a, b)| a == b).count()
        };

        let quad_volume_overlap = |a: Quad, b: Quad| -> f64 {
            let tet_a = na::Matrix3x4::from_columns(&a.map(particle_position));
            let tet_b = na::Matrix3x4::from_columns(&b.map(particle_position));
            if crate::geometry::tetrahedra_overlap(&tet_a, &tet_b) {
                crate::geometry::approximate_tetrahedron_overlap_volume(&tet_a, &tet_b)
            } else {
                0.0
            }
        };

        if quads.len() > 100 {
            // Too many quads, limit to two-vertex overlap
            let mut limited = Vec::new();
            for q in quads {
                if limited.iter().all(|&p| quad_vertex_overlap(p, q) <= 2) {
                    limited.push(q);
                }
            }
            quads = limited;
        }

        // Find selection of quads that covers all vertices with minimal overlap
        let q = quads.len();
        let min_k = (n as f32 / 4.0).ceil() as usize;
        let (minimal_covering_quads, _) = (min_k..=q)
            .map(|i| quads.iter().combinations(i))
            .find_map(|combinations| combinations.fold(
                None,
                |maybe_best, quad_combination| {
                    // Test cover, and return early if doesn't cover
                    let mut covered_particles = quad_combination.iter().flat_map(|q| q.iter()).map(|&&p| p).collect::<HashSet<Particle>>();
                    covered_particles.remove(&Particle::Origin);
                    let covers = covered_particles.len() == n;
                    if !covers {
                        return maybe_best;
                    }

                    let vertex_overlap: usize = quad_combination.iter()
                        .tuple_combinations()
                        .map(|(&i, &j)| quad_vertex_overlap(*i, *j))
                        .sum();

                    let volume_overlap: f64 = quad_combination.iter()
                        .tuple_combinations()
                        .map(|(&i, &j)| quad_volume_overlap(*i, *j))
                        .sum();

                    let overlap = (vertex_overlap, volume_overlap);

                    if let Some((best_combination, best_overlap)) = maybe_best {
                        if overlap < best_overlap {
                            Some((quad_combination, overlap))
                        } else {
                            Some((best_combination, best_overlap))
                        }
                    } else {
                        Some((quad_combination, overlap))
                    }
                }
            )).expect("Can always find a covering combination");

        // Un-ref particles by copying
        minimal_covering_quads.into_iter().map(|arr| arr.map(|&p| p)).collect()
    }

    /// Reorder coordinates into a canonical vertex order
    pub fn canonicalize_coordinates(coords: Matrix3N) -> Matrix3N {
        let (top, mut reoriented) = crate::inertia::standardize_top(coords);
        let n = reoriented.ncols();

        let rotation_basis = Self::find_rotation_basis(&reoriented);
        let vertex_groups = {
            let mut sets = UnionFind::new(n);
            for rotation in rotation_basis.iter() {
                for v in 0..n {
                    sets.union(v, rotation.permutation[v]);
                }
            }
            Self::union_to_groups(sets)
        };

        if top == crate::inertia::Top::Spherical {
            // If there's a unique vertex (by rotation), that one should be +z
            let maybe_unique = vertex_groups.iter()
                .find_map(|group| (group.len() == 1).then(|| group[0]));

            if let Some(unique_vertex) = maybe_unique {
                let rotation = na::Rotation::<f64, 3>::rotation_between(
                    &reoriented.column(unique_vertex.get()),
                    &na::Vector3::z_axis()
                ).expect("Can generate rotation for unique vertex");
                reoriented = rotation * reoriented;
            }
        }

        // Group vertices by planes along z 
        let z_plane_groups = {
            let mut sets = UnionFind::new(n);
            for (i, j) in (0..n).tuple_combinations() {
                let delta_z = reoriented.column(j).z - reoriented.column(i).z;
                if delta_z.abs() < 1e-2 {
                    sets.union(i, j);
                }
            }
            let groups = Self::union_to_groups(sets);

            // And sort by z value
            let zs: Vec<_> = groups.iter()
                .map(|group| {
                    let z = reoriented.column(group.first().unwrap().get()).z;
                    NotNan::new(z).expect("Not NaN")
                })
                .collect();
            Permutation::ordering(&zs).apply_move(groups).expect("ordering produces valid permutation")
        };

        let vertex_color: HashMap<Vertex, usize> = vertex_groups.iter()
            .enumerate()
            .flat_map(|(i, group)| group.iter().map(move |&v| (v, i)))
            .collect();

        // Find a z-plane that contains the smallest unique off-axis vertex group
        let pivot_plane = z_plane_groups.iter()
            .filter(|plane_vertices| {
                if plane_vertices.len() == 1 {
                    let z_dist = crate::geometry::axis_distance(
                        &na::Vector3::z_axis(),
                        &reoriented.column(plane_vertices[0].get())
                    );

                    if z_dist < 1e-4 {
                        return false;
                    }
                }

                true

            })
            .min_by_key(|plane_vertices| {
                let num_colors = plane_vertices.iter()
                    .map(|v| vertex_color[v])
                    .unique()
                    .count();

                (plane_vertices.len(), num_colors)
            });

        // There is a pivot plane if there is a single vertex off z-axis
        let pivot_phi = pivot_plane.map(|vertices| {
            // Pick any vertex in the pivot plane (dubious)
            let v = reoriented.column(vertices[0].get());
            (v.x / (v.x.powi(2) + v.y.powi(2)).sqrt()).acos().copysign(v.y)
        });

        let mut ordering = vec![0; n];
        let mut label = 0;

        for mut plane_vertices in z_plane_groups {
            if plane_vertices.len() == 1 {
                ordering[plane_vertices[0].get()] = label;
                label += 1;
            } else {
                plane_vertices.sort_by_key(|vertex| {
                    let v = reoriented.column(vertex.get());
                    let phi = (v.x / (v.x.powi(2) + v.y.powi(2)).sqrt()).acos().copysign(v.y);
                    if phi < pivot_phi.expect("Pivot exists") - std::f64::consts::TAU / 360.0 {
                        NotNan::new(phi + std::f64::consts::TAU).unwrap()
                    } else {
                        NotNan::new(phi).unwrap()
                    }
                });

                for v in plane_vertices {
                    ordering[v.get()] = label;
                    label += 1;
                }
            }
        }

        let ordering = Permutation::try_from(ordering)
            .expect("Generated valid permutation")
            .inverse();

        apply_permutation(&reoriented, &ordering)
    }
}

pub mod statics;
pub use statics::*;

/// Look up a shape from its name
pub fn shape_from_name(name: Name) -> &'static Shape {
    let shape = SHAPES[name as usize];
    assert_eq!(shape.name, name);
    shape
}

pub mod similarity;
pub mod recognition;

#[cfg(test)]
mod tests {
    use crate::shapes::*;

    fn vertex_group_correct(shape: &Shape, expected_len: usize) {
        let vertex_groups = shape.vertex_groups();
        assert_eq!(vertex_groups.len(), expected_len);

        // Every vertex occurs once
        let mut counts = Vec::new();
        counts.resize(shape.num_vertices(), 0);
        for group in vertex_groups.iter() {
            for v in group.iter() {
                let usize_v = v.get();
                assert!(usize_v < shape.num_vertices());
                counts[usize_v] += 1;
            }
        }

        assert!(counts.iter().all(|c| *c == 1));
    }

    #[test]
    fn vertex_groups() {
        vertex_group_correct(&TSHAPE, 2);
        vertex_group_correct(&SQUARE, 1);
        vertex_group_correct(&TETRAHEDRON, 1);
        vertex_group_correct(&SEESAW, 2);
        vertex_group_correct(&SQUAREPYRAMID, 2);
        vertex_group_correct(&OCTAHEDRON, 1);
        vertex_group_correct(&PENTAGONALPYRAMID, 2);
    }

    #[test]
    fn names_correct() {
        assert!(SHAPES.iter().all(|s| std::ptr::eq(shape_from_name(s.name), *s)));
    }

    #[test]
    fn is_rotation_works() {
        let tetr_rotations = TETRAHEDRON.generate_rotations();
        let occupation: Bijection<Vertex, Column> = Bijection::new_random(TETRAHEDRON.num_vertices());
        for rot in &tetr_rotations {
            let rotated_occupation = rot.compose(&occupation).expect("fine");
            assert!(Shape::is_rotation(&occupation, &rotated_occupation, &tetr_rotations));
        }
    }

    #[test]
    fn canonical_vertex_orders() {
        for shape in SHAPES.iter() {
            assert!(
                Shape::can_canonicalize(&shape.coordinates),
                "Shape {} cannot be canonicalized",
                shape.name
            );
        }
    }
}
