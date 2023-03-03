// TODO
// - Expand shape data

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
    Hexagon
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
        }
    }
}

impl std::fmt::Display for Name {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.repr())
    }
}

use crate::permutation::Permutation;
use crate::strong::bijection::Bijection;

#[derive(Index, Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Vertex(u8);

type Rotation = Bijection<Vertex, Vertex>;
type Mirror = Bijection<Vertex, Vertex>;

#[derive(Index, Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Column(u8);

pub static ORIGIN: Vertex = Vertex(u8::MAX);

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

    pub fn is_rotation<T: NewTypeIndex>(&self, a: &Bijection<Vertex, T>, b: &Bijection<Vertex, T>, rotations: &HashSet<Rotation>) -> bool {
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

    pub fn vertex_groups_holding(&self, held: Vertex, rotations: &HashSet<Rotation>) -> Vec<Vec<Vertex>> {
        let shape_size = self.size();
        let mut sets = UnionFind::new(shape_size);
        for rotation in rotations.iter().filter(|rot| rot.get(&held) == Some(held)) {
            for v in 0..shape_size {
                sets.union(v, rotation.permutation[v] as usize);
            }
        }

        Self::union_to_groups(sets)
    }
}

fn make_rotation(slice: &[u8]) -> Rotation {
    Rotation::new(Permutation {sigma: slice.to_vec()})
}

fn make_mirror(slice: &[u8]) -> Option<Mirror> {
    Some(make_rotation(slice))
}

lazy_static! {
    pub static ref LINE: Shape = Shape {
        name: Name::Line,
        coordinates: Matrix3N::from_column_slice(&[
             1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0
        ]),
        rotation_basis: vec![make_rotation(&[1, 0])],
        tetrahedra: vec![],
        mirror: None
    };

    /// Bent at 107°
    pub static ref BENT: Shape = Shape {
        name: Name::Bent,
        coordinates: Matrix3N::from_column_slice(&[
            1.0, 0.0, 0.0,
            -0.292372, 0.956305, 0.0
        ]),
        rotation_basis: vec![make_rotation(&[1, 0])],
        tetrahedra: vec![],
        mirror: None
    };

    pub static ref EQUILATERAL_TRIANGLE: Shape = Shape {
        name: Name::EquilateralTriangle,
        coordinates: Matrix3N::from_column_slice(&[
            1.0, 0.0, 0.0,
            -0.5, 0.866025, 0.0,
            -0.5, -0.866025, 0.0
        ]),
        rotation_basis: vec![
            make_rotation(&[1, 2, 0]),
            make_rotation(&[0, 2, 1])
        ],
        tetrahedra: vec![],
        mirror: None
    };

    /// Monovacant tetrahedron. 
    ///
    /// Widely called trigonal pyramidal, but easily confusable with a 
    /// face-centered trigonal pyramid.
    pub static ref VACANT_TETRAHEDRON: Shape = Shape {
        name: Name::VacantTetrahedron,
        coordinates: Matrix3N::from_column_slice(&[
            0.0, -0.366501, 0.930418,
            0.805765, -0.366501, -0.465209,
            -0.805765, -0.366501, -0.465209
        ]),
        rotation_basis: vec![make_rotation(&[2, 0, 1])],
        tetrahedra: vec![[ORIGIN, Vertex(0), Vertex(1), Vertex(2)]],
        mirror: make_mirror(&[0, 2, 1])
    };

    pub static ref TSHAPE: Shape = Shape {
        name: Name::T,
        coordinates: Matrix3N::from_column_slice(&[
            -1.0, -0.0, -0.0,
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
        ]),
        rotation_basis: vec![Rotation::new(Permutation {sigma: vec![2, 1, 0]})],
        tetrahedra: vec![],
        mirror: None
    };

    pub static ref TETRAHEDRON: Shape = Shape {
        name: Name::Tetrahedron,
        coordinates: Matrix3N::from_column_slice(&[
            0.0, 1.0, 0.0,
            0.0, -0.333807, 0.942641,
            0.816351, -0.333807, -0.471321,
            -0.816351, -0.333807, -0.471321
        ]),
        rotation_basis: vec![
            make_rotation(&[0, 3, 1, 2]),
            make_rotation(&[2, 1, 3, 0]),
            make_rotation(&[3, 0, 2, 1]),
            make_rotation(&[1, 2, 0, 3])
        ],
        tetrahedra: vec![[Vertex(0), Vertex(1), Vertex(2), Vertex(3)]],
        mirror: make_mirror(&[0, 2, 1, 3])
    };

    pub static ref SQUARE: Shape = Shape {
        name: Name::Square,
        coordinates: Matrix3N::from_column_slice(&[
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            -1.0, -0.0, -0.0,
            -0.0, -1.0, -0.0
        ]),
        rotation_basis: vec![
            make_rotation(&[3, 0, 1, 2]),
            make_rotation(&[1, 0, 3, 2]),
            make_rotation(&[3, 2, 1, 0]),
        ],
        tetrahedra: vec![],
        mirror: None
    };

    /// Equatorially monovacant trigonal bipyramid or edge-centered tetragonal disphenoid
    pub static ref SEESAW: Shape = Shape {
        name: Name::Seesaw,
        coordinates: Matrix3N::from_column_slice(&[
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            -0.5, 0.0, -0.866025,
            -0.0, -1.0, -0.0
        ]),
        rotation_basis: vec![make_rotation(&[3, 2, 1, 0])],
        tetrahedra: vec![
            [Vertex(0), ORIGIN, Vertex(1), Vertex(2)],
            [ORIGIN, Vertex(3), Vertex(1), Vertex(2)]
        ],
        mirror: make_mirror(&[0, 2, 1, 3])
    };

    /// Face-centered trigonal pyramid = trig. pl. + axial ligand 
    /// (or monovacant trigonal bipyramid)
    pub static ref TRIGONALPYRAMID: Shape = Shape {
        name: Name::TrigonalPyramid,
        coordinates: Matrix3N::from_column_slice(&[
            1.0, 0.0, 0.0,
            -0.5, 0.866025, 0.0,
            -0.5, -0.866025, 0.0,
            0.0, 0.0, 1.0
        ]),
        rotation_basis: vec![make_rotation(&[2, 0, 1, 3])],
        tetrahedra: vec![[Vertex(0), Vertex(1), Vertex(3), Vertex(2)]],
        mirror: make_mirror(&[0, 2, 1, 3])
    };

    /// J1 solid (central position is square-face centered)
    pub static ref SQUAREPYRAMID: Shape = Shape {
        name: Name::SquarePyramid,
        coordinates: Matrix3N::from_column_slice(&[
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            -1.0, 0.0, 0.0,
            0.0, -1.0, 0.0,
            0.0, 0.0, 1.0,
        ]),
        rotation_basis: vec![make_rotation(&[3, 0, 1, 2, 4])],
        tetrahedra: vec![
            [Vertex(0), Vertex(1), Vertex(4), ORIGIN],
            [Vertex(1), Vertex(2), Vertex(4), ORIGIN],
            [Vertex(2), Vertex(3), Vertex(4), ORIGIN],
            [Vertex(3), Vertex(0), Vertex(4), ORIGIN],
        ],
        mirror: make_mirror(&[1, 0, 3, 2, 4])
    };

    /// J12 solid
    pub static ref TRIGONALBIPYRAMID: Shape = Shape {
        name: Name::TrigonalBipyramid,
        coordinates: Matrix3N::from_column_slice(&[
            1.0, 0.0, 0.0,
            -0.5, 0.866025, 0.0,
            -0.5, -0.866025, 0.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, -1.0
        ]),
        rotation_basis: vec![
            make_rotation(&[2, 0, 1, 3, 4]), // C3
            make_rotation(&[0, 2, 1, 4, 3]), // C2 on 0
        ],
        tetrahedra: vec![
            [Vertex(0), Vertex(1), Vertex(3), Vertex(2)], 
            [Vertex(0), Vertex(1), Vertex(2), Vertex(4)]
        ],
        mirror: make_mirror(&[0, 2, 1, 3, 4])
    };

    pub static ref PENTAGON: Shape = Shape {
        name: Name::Pentagon,
        coordinates: Matrix3N::from_column_slice(&[
            1.0, 0.0, 0.0,
            0.309017, 0.951057, 0.0,
            -0.809017, 0.587785, 0.0,
            -0.809017, -0.587785, 0.0,
            0.309017, -0.951057, 0.0
        ]),
        rotation_basis: vec![
            make_rotation(&[4, 0, 1, 2, 3]),
            make_rotation(&[0, 4, 3, 2, 1]),
        ],
        tetrahedra: vec![],
        mirror: None
    };

    pub static ref OCTAHEDRON: Shape = Shape {
        name: Name::Octahedron,
        coordinates: Matrix3N::from_column_slice(&[
            1.0,  0.0,  0.0,
            0.0,  1.0,  0.0,
           -1.0,  0.0,  0.0,
            0.0, -1.0,  0.0,
            0.0,  0.0,  1.0,
            0.0,  0.0, -1.0,
        ]),
        rotation_basis: vec![
            make_rotation(&[3, 0, 1, 2, 4, 5]),
            make_rotation(&[0, 5, 2, 4, 1, 3]),
            make_rotation(&[4, 1, 5, 3, 2, 0]), // TODO maybe unnecessary?
        ],
        tetrahedra: vec![ // TODO check if reducible
            [Vertex(3), Vertex(0), Vertex(4), ORIGIN],
            [Vertex(0), Vertex(1), Vertex(4), ORIGIN],
            [Vertex(1), Vertex(2), Vertex(4), ORIGIN],
            [Vertex(2), Vertex(3), Vertex(4), ORIGIN],
            [Vertex(3), Vertex(0), ORIGIN, Vertex(5)],
            [Vertex(0), Vertex(1), ORIGIN, Vertex(5)],
            [Vertex(1), Vertex(2), ORIGIN, Vertex(5)],
            [Vertex(2), Vertex(3), ORIGIN, Vertex(5)],
        ],
        mirror: make_mirror(&[1, 0, 3, 2, 4, 5])
    };

    pub static ref TRIGONALPRISM: Shape = Shape {
        name: Name::TrigonalPrism,
        coordinates: Matrix3N::from_column_slice(&[
             0.755929,  0.000000,  0.654654,
            -0.377964,  0.654654,  0.654654,
            -0.377964, -0.654654,  0.654654,
             0.755929,  0.000000, -0.654654,
            -0.377964,  0.654654, -0.654654,
            -0.377964, -0.654654, -0.654654
        ]),
        rotation_basis: vec![
            make_rotation(&[2, 0, 1, 5, 3, 4]), // C3 axial
            make_rotation(&[3, 5, 4, 0, 2, 1]), // C2 between 0, 3
        ],
        tetrahedra: vec![
            [ORIGIN, Vertex(0), Vertex(2), Vertex(1)],
            [Vertex(3), ORIGIN, Vertex(5), Vertex(4)]
        ],
        mirror: make_mirror(&[0, 2, 1, 3, 5, 4])
    };
    
    /// J2 solid
    pub static ref PENTAGONALPYRAMID: Shape = Shape {
        name: Name::PentagonalPyramid,
        coordinates: Matrix3N::from_column_slice(&[
            1.0, 0.0, 0.0,
            0.309017, 0.951057, 0.0,
            -0.809017, 0.587785, 0.0,
            -0.809017, -0.587785, 0.0,
            0.309017, -0.951057, 0.0,
            0.0, 0.0, 1.0
        ]),
        rotation_basis: vec![
            make_rotation(&[4, 0, 1, 2, 3, 5]),
        ],
        tetrahedra: vec![
            [Vertex(0), ORIGIN, Vertex(1), Vertex(5)],
            [Vertex(1), ORIGIN, Vertex(2), Vertex(5)],
            [Vertex(2), ORIGIN, Vertex(3), Vertex(5)],
            [Vertex(3), ORIGIN, Vertex(4), Vertex(5)],
            [Vertex(4), ORIGIN, Vertex(0), Vertex(5)],

        ],
        mirror: make_mirror(&[0, 4, 3, 2, 1, 5])
    };

    pub static ref HEXAGON: Shape = Shape {
        name: Name::Hexagon,
        coordinates: Matrix3N::from_column_slice(&[
             1.000000,  0.000000,  0.000000,
             0.500000,  0.866025,  0.000000,
            -0.500000,  0.866025,  0.000000,
            -1.000000,  0.000000,  0.000000,
            -0.500000, -0.866025,  0.000000,
             0.500000, -0.866025,  0.000000
        ]),
        rotation_basis: vec![
            make_rotation(&[5, 0, 1, 2, 3, 4]),
            make_rotation(&[0, 5, 4, 3, 2, 1]),
        ],
        tetrahedra: vec![],
        mirror: None
    };

    pub static ref SHAPES: Vec<&'static Shape> = vec![&LINE, &BENT, &EQUILATERAL_TRIANGLE, &VACANT_TETRAHEDRON, &TSHAPE, &TETRAHEDRON, &SQUARE, &SEESAW, &TRIGONALPYRAMID, &SQUAREPYRAMID, &TRIGONALBIPYRAMID, &PENTAGON, &OCTAHEDRON, &TRIGONALPRISM, &PENTAGONALPYRAMID, &HEXAGON];
}

pub fn shape_from_name(name: Name) -> &'static Shape {
    let shape = SHAPES[name as usize];
    assert_eq!(shape.name, name);
    shape
}

pub mod similarity;

#[cfg(test)]
mod tests {
    use crate::shapes::*;
    use crate::strong::matrix::AsNewTypeIndexedMatrix;

    fn tetrahedron_volume(tetrahedron: &[Vertex; 4], points: &Matrix3N) -> f64 {
        let coords = AsNewTypeIndexedMatrix::<Vertex>::new(points);
        let zero = Matrix3N::zeros(1);
        let r = |v: Vertex| {
            if v == ORIGIN {
                zero.column(0)
            } else {
                coords.column(v)
            }
        };

        (r(tetrahedron[0]) - r(tetrahedron[3])).dot(
            &(r(tetrahedron[1]) - r(tetrahedron[3])).cross(
                &(r(tetrahedron[2]) - r(tetrahedron[3]))
            )
        )
    }

    #[test]
    fn all_tetrahedra_positive_volume() {
        for shape in SHAPES.iter() {
            let mut pass = true;
            for tetrahedron in shape.tetrahedra.iter() {
                let volume = tetrahedron_volume(&tetrahedron, &shape.coordinates);
                if volume < 0.0 {
                    pass = false;
                    println!("Shape {} tetrahedron {:?} does not have positive volume.", shape.name, tetrahedron);
                }
            }
            assert!(pass);
        }
    }

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
}
