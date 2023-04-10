extern crate nalgebra as na;
type Matrix3N = na::Matrix3xX<f64>;
type Vector3 = na::Vector3<f64>;

use std::collections::{HashSet, HashMap};
use itertools::Itertools;
use petgraph::unionfind::UnionFind;

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
        match self {
            Name::Line => "line",
            Name::Bent => "bent",
            Name::EquilateralTriangle => "triangle",
            Name::VacantTetrahedron => "vacant tetrahedron",
            Name::T => "T-shaped",
            Name::Tetrahedron => "tetrahedron",
            Name::Square => "square",
            Name::Seesaw => "seesaw",
            Name::TrigonalPyramid => "trigonal pyramid",
            Name::SquarePyramid => "square pyramid",
            Name::TrigonalBipyramid => "trigonal bipyramid",
            Name::Pentagon => "pentagon",
            Name::Octahedron => "octahedron",
            Name::TrigonalPrism => "trigonal prism",
            Name::PentagonalPyramid => "pentagonal pyramid",
            Name::Hexagon => "hexagon",
            Name::PentagonalBipyramid => "pentagonal bipyramid",
            Name::CappedOctahedron => "capped octahedron",
            Name::CappedTrigonalPrism => "capped trigonal prism",
            Name::SquareAntiprism => "square antiprism",
            Name::Cube => "cube",
            Name::TrigonalDodecahedron => "trigonal dodecahedron",
            Name::HexagonalBipyramid => "hexagonal bipyramid",
            Name::TricappedTrigonalPrism => "tricapped trigonal prism",
            Name::CappedSquareAntiprism => "capped square antiprism",
            Name::HeptagonalBipyramid => "heptagonal bipyramid",
            Name::BicappedSquareAntiprism => "bicapped square antiprism",
            Name::EdgeContractedIcosahedron => "edge contracted icosahedron",
            Name::Icosahedron => "icosahedron",
            Name::Cuboctahedron => "cuboctahedron",
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

#[derive(Index, Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Vertex(u8);

type Rotation = Bijection<Vertex, Vertex>;
type Mirror = Bijection<Vertex, Vertex>;

#[derive(Index, Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Column(u8);

pub struct Shape {
    pub name: Name,
    /// Unit sphere coordinates without a centroid
    pub coordinates: Matrix3N,
    /// Spatial rotational basis expressed by vertex permutations
    pub rotation_basis: Vec<Rotation>,
    /// Minimal set of tetrahedra required to distinguish volumes in DG
    pub tetrahedra: Vec<[Vertex; 4]>,
    /// Mirror symmetry element expressed by vertex permutation, if present
    pub mirror: Option<Mirror>
}

pub struct VertexPlane {
    pub vertices: Vec<usize>,
    pub plane: Plane
}

impl Shape {
    /// Number of vertices of the shape
    pub fn size(&self) -> usize {
        self.coordinates.ncols()
    }

    fn expand_rotation_basis(&self, basis: &[Rotation]) -> HashSet<Rotation> {
        let mut rotations: HashSet<Rotation> = HashSet::new();
        rotations.insert(Rotation::identity(self.size()));
        let max_basis_idx = basis.len() - 1;

        struct Frame {
            permutation: Rotation,
            next_basis: usize
        }

        let mut stack = Vec::<Frame>::new();
        stack.push(Frame {permutation: Rotation::identity(self.size()), next_basis: 0});

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
        self.expand_rotation_basis(self.rotation_basis.as_slice())
    }

    pub fn is_rotation<T: NewTypeIndex>(a: &Bijection<Vertex, T>, b: &Bijection<Vertex, T>, rotations: &HashSet<Rotation>) -> bool {
        rotations.iter().any(|r| r.compose(a).expect("Bad occupations") == *b)
    }

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
            groups[*g].push(Vertex::from(v as u8));
        }

        groups
    }

    pub fn vertex_groups(&self) -> Vec<Vec<Vertex>> {
        let shape_size = self.size();
        let mut sets = UnionFind::new(shape_size);
        for rotation in self.rotation_basis.iter() {
            for v in 0..shape_size {
                sets.union(v, rotation.permutation[v] as usize);
            }
        }

        Self::union_to_groups(sets)
    }

    pub fn vertex_groups_holding(&self, held: &[Vertex], rotations: &HashSet<Rotation>) -> Vec<Vec<Vertex>> {
        let shape_size = self.size();
        let mut sets = UnionFind::new(shape_size);
        for rotation in rotations.iter().filter(|rot| held.iter().all(|v| rot.is_fixed_point(*v))) {
            for v in 0..shape_size {
                sets.union(v, rotation.permutation[v] as usize);
            }
        }

        Self::union_to_groups(sets)
    }

    /// Collect coplanar vertices and their joint plane
    fn coplanar_vertex_groups(&self) -> HashMap<Vec<usize>, Plane> {
        let mut groups = HashMap::new();

        let n = self.size();
        for mut vertex_triple in (0..n).combinations(3) {
            let mut plane = Plane::fit_matrix_points(&self.coordinates, &vertex_triple);
            let mut coplanar_vertices: Vec<usize> = (0..n)
                .filter(|v| !vertex_triple.contains(v) && plane.signed_distance(&self.coordinates.column(*v)).abs() < 1e-3)
                .collect();

            if !coplanar_vertices.is_empty() {
                coplanar_vertices.append(&mut vertex_triple);
                coplanar_vertices.sort();
                let mut updated_centroid = Vector3::zeros();
                for &v in coplanar_vertices.iter() {
                    updated_centroid += self.coordinates.column(v);
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
    fn parallel_vertex_planes(&self) -> Vec<Vec<VertexPlane>> {
        let groups = self.coplanar_vertex_groups();
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
    fn permutation_is_rotation(&self, permutation: &Permutation) -> bool {
        let normalized_points = unit_sphere_normalize(self.coordinates.clone());
        let permuted = unit_sphere_normalize(apply_permutation(&normalized_points, permutation));
        let fit = crate::quaternions::fit(&normalized_points, &permuted);
        fit.msd < 1e-6
    }

    /// Finds proper rotations containing at least one coplanar set of at least three
    /// vertices. I.e. can find rotations of order two if it also happens to rotate a coplanar
    /// vertex group of four vertices, but cannot find all rotations of order two generally. Mostly
    /// finds rotations of order three and above.
    fn try_parallel_vertex_plane_axis(&self, planes: &[VertexPlane]) -> Option<Permutation> {
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
        let n = self.size();
        let mut remaining_vertices: Vec<usize> = (0..n)
            .filter(|v| !planes.iter().any(|p| p.vertices.contains(v)))
            .collect();

        // TODO remove axial vertices first? Fails faster

        // If the minimal order is two, look for axis-perpendicular dipoles
        // Those are neither in the plane vertex triples nor are axial
        let mut dipoles: Vec<(usize, usize)> = Vec::new();
        if minimal_order % 2 == 0 {
            'outer: for (i, &a) in remaining_vertices.iter().enumerate() {
                let component_a = axis_perpendicular_component(&axis, &self.coordinates.column(a));
                if component_a.norm_squared() < 1e-3 {
                    continue;
                }

                for &b in remaining_vertices.iter().skip(i + 1) {
                    let component_b = axis_perpendicular_component(&axis, &self.coordinates.column(b));
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
            .all(|v| axis_distance(&axis, &self.coordinates.column(*v)) < 1e-3);

        if !remainder_axial {
            return None;
        }

        // Try lowest order rotation (proper and improper)
        let mut proper = Permutation::identity(n);
        for p in planes.iter() {
            let anchor = p.vertices.first().expect("Guaranteed three vertices per group");
            let positive_signed_angle = |v| {
                let a = self.coordinates.column(*anchor);
                let b = self.coordinates.column(v);
                let mut angle = a.cross(&b).dot(&axis).atan2(a.dot(&b));
                if angle < 0.0 {
                    angle += std::f64::consts::TAU;
                }
                angle
            };
            let mut with_signed_angles: Vec<(usize, f64)> = p.vertices.iter().skip(1)
                .map(|&v| (v, positive_signed_angle(v))).collect();
            with_signed_angles.sort_by(|(_, alpha), (_, beta)| alpha.partial_cmp(beta).expect("No NaN angles"));

            let mut ordered_vertices: Vec<usize> = with_signed_angles.into_iter()
                .map(|(v, _)| v)
                .collect();
            ordered_vertices.push(*anchor);

            for (&i, &j) in ordered_vertices.iter().circular_tuple_windows() {
                proper.sigma[i] = j as u8;
            }
        }
        for (i, j) in dipoles.into_iter() {
            proper.sigma[i] = j as u8;
            proper.sigma[j] = i as u8;
        }

        // TODO maybe unnecessary to test
        if self.permutation_is_rotation(&proper) && proper.index() != 0 {
            Some(proper)
        } else {
            None
        }
    }

    /// Test whether a vector is a suitable c2 axis
    fn try_c2_axis(&self, axis: Vector3) -> Option<Permutation> {
        let rotated = nalgebra::Rotation3::from_axis_angle(
            &nalgebra::Unit::new_normalize(axis),
            std::f64::consts::PI
        ) * self.coordinates.clone();

        let n = self.size();
        let mut proper = Permutation::identity(n);
        'outer: for i in 0..n {
            if proper[i] != i as u8 {
                continue;
            }

            for j in i..n {
                if proper[j] != j as u8 {
                    continue;
                }

                let distance = (rotated.column(i) - self.coordinates.column(j)).norm();
                if distance < 0.1 {
                    proper.sigma[i] = j as u8;
                    proper.sigma[j] = i as u8;

                    continue 'outer;
                }
            }

            return None;
        }

        if proper.index() != 0 && self.permutation_is_rotation(&proper) {
            Some(proper)
        } else {
            None
        }
    }

    /// Try to find all c2 axes in a shape
    fn find_c2_axes(&self) -> Vec<Permutation> {
        let mut rotations: Vec<Permutation> = Vec::new();
        for (i, col_i) in self.coordinates.column_iter().enumerate() {
            // Try each particle position
            if let Some(rotation) = self.try_c2_axis(col_i.into()) {
                rotations.push(rotation);
            }

            // Try each sum of two particle positions, if they don't cancel
            for col_j in self.coordinates.column_iter().skip(i + 1) {
                let sum = col_i + col_j;
                // Canceling vectors give wild new normalized points, avoid:
                if sum.norm_squared() > 1e-6 {
                    if let Some(rotation) = self.try_c2_axis(sum.normalize()) {
                        rotations.push(rotation);
                    }
                }
            }
        }

        rotations
    }

    /// Reduce a set of rotations to a basis that generates all
    fn reduce_to_basis(&self, mut rotations: HashSet<Rotation>) -> Vec<Rotation> {
        // Find minimal rotations necessary to generate the most rotations
        let min_fixed_rot = rotations.iter()
            .max_by_key(|r| r.permutation.cycles().iter().map(|c| c.len()).sum::<usize>())
            .unwrap()
            .clone();
        rotations.remove(&min_fixed_rot);
        let mut basis = vec![min_fixed_rot];
        let mut basis_size = self.expand_rotation_basis(basis.as_slice()).len();

        loop {
            let maybe_next_best = rotations.iter()
                .fold(
                    None,
                    |best, r| {
                        basis.push(r.clone());
                        let new_basis_size = self.expand_rotation_basis(basis.as_slice()).len();
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
    fn stupid_find_rotations(&self) -> Vec<Rotation> {
        // TODO this is really very stupid, as 
        // - repeated applications of true rotations aren't excluded / skipped
        // - combinations of found rotations aren't excluded / skipped
        permutations(self.size())
            .filter(|p| self.permutation_is_rotation(p))
            .map(Rotation::new)
            .collect()
    }

    /// Find rotation basis of a shape
    pub fn find_rotation_basis(&self) -> Vec<Rotation> {
        let plane_axes: Vec<Permutation> = self.parallel_vertex_planes().into_iter()
            .filter_map(|v| self.try_parallel_vertex_plane_axis(v.as_slice()))
            .collect();
        let c2_axes = self.find_c2_axes();

        // plane axes and c2 axes may have overlapping rotations, avoid duplicates
        let mut rotations = plane_axes.into_iter()
            .chain(c2_axes.into_iter())
            .map(Rotation::new)
            .collect::<HashSet<Rotation>>();

        if rotations.is_empty() {
            if self.size() > 4 {
                panic!("No rotations found heuristically for {}, too large to use exhaustive search", self.name);
            }

            rotations = HashSet::from_iter(self.stupid_find_rotations().into_iter());
        }

        self.reduce_to_basis(rotations)
    }

    pub fn find_mirror(&self) -> Option<Mirror> {
        let inverted = -1.0 * self.coordinates.clone();

        // if the quaternion fit onto the original coordinates fails, there is a mirror element
        // the mirror is the permutation from similarity-fitting the inverted coordinates onto the
        // original ones
        //
        // TODO Is there a better (faster) way?
        // TODO this fails because a centroid is expected :(
        let similarity = similarity::polyhedron(&inverted, self.name).ok()?;
        if dbg!(similarity.csm) < 1e-6 {
            Some(Mirror::new(similarity.bijection.permutation))
        } else {
            None
        }
    }
}

pub mod statics;
pub use statics::*;

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
        counts.resize(shape.size(), 0);
        for group in vertex_groups.iter() {
            for v in group.iter() {
                let usize_v = v.get() as usize;
                assert!(usize_v < shape.size());
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
        let occupation: Bijection<Vertex, Column> = Bijection::from_index(4, 23);
        for rot in &tetr_rotations {
            let rotated_occupation = rot.compose(&occupation).expect("fine");
            assert!(Shape::is_rotation(&occupation, &rotated_occupation, &tetr_rotations));
        }
    }
}
