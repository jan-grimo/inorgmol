extern crate nalgebra as na;
type Matrix3N = na::Matrix3xX<f64>;

use std::collections::{HashSet, HashMap};

use crate::strong::{Index, NewTypeIndex};
use petgraph::unionfind::UnionFind;

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
    Cuboctahedron
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
            Name::Cuboctahedron => "cuboctahedron"
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

impl Shape {
    /// Number of vertices of the shape
    pub fn size(&self) -> usize {
        self.coordinates.ncols()
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
        let mut rotations: HashSet<Rotation> = HashSet::new();
        rotations.insert(Rotation::identity(self.size()));
        let max_basis_idx = self.rotation_basis.len() - 1;

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
            let next_rotation = &self.rotation_basis[latest.next_basis];
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
}

pub mod statics;
pub use statics::*;

pub fn shape_from_name(name: Name) -> &'static Shape {
    let shape = SHAPES[name as usize];
    assert_eq!(shape.name, name);
    shape
}

pub mod similarity;

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
}
