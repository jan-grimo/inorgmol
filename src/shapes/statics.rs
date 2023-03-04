use crate::shapes::{Shape, Name, Vertex, Rotation, Mirror, Matrix3N};
use crate::permutation::Permutation;

fn make_rotation(slice: &[u8]) -> Rotation {
    Rotation::new(Permutation {sigma: slice.to_vec()})
}

fn make_mirror(slice: &[u8]) -> Option<Mirror> {
    Some(make_rotation(slice))
}

pub static ORIGIN: Vertex = Vertex(u8::MAX);

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

    /// J13 solid
    pub static ref PENTAGONALBIPYRAMID: Shape = Shape {
        name: Name::PentagonalBipyramid,
        coordinates: Matrix3N::from_column_slice(&[
            1.0, 0.0, 0.0,
            0.309017, 0.951057, 0.0,
            -0.809017, 0.587785, 0.0,
            -0.809017, -0.587785, 0.0,
            0.309017, -0.951057, 0.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, -1.0
        ]),
        rotation_basis: vec![
            make_rotation(&[4, 0, 1, 2, 3, 5, 6]),
            make_rotation(&[1, 0, 4, 3, 2, 6, 5]),
        ],
        tetrahedra: vec![
            [Vertex(0), Vertex(1), Vertex(5), Vertex(6)],
            [Vertex(1), Vertex(2), Vertex(5), Vertex(6)],
            [Vertex(2), Vertex(3), Vertex(5), Vertex(6)],
            [Vertex(3), Vertex(4), Vertex(5), Vertex(6)],
            [Vertex(4), Vertex(0), Vertex(5), Vertex(6)],

        ],
        mirror: make_mirror(&[0, 4, 3, 2, 1, 5, 6])
    };

    /// Capped octahedron
    ///
    /// This is a gyroelongated triangular pyramid, or alternatively a 
    /// "capped triangular antiprism", depending on whatever helps you 
    /// visualize it.
    ///
    /// Coordinates a spherized [V(CO)7]+ in C3v (find local minimium of 
    /// Thomson potential)
    ///
    /// from Jay W. Dicke, Nathan J. Stibrich, Henry F. Schaefer,
    /// V(CO)7+: A capped octahedral structure completes the 18-electron rule,
    /// Chemical Physics Letters, Volume 456, Issues 1–3, 2008.
    ////
    pub static ref CAPPEDOCTAHEDRON: Shape = Shape {
        name: Name::CappedOctahedron,
        coordinates: Matrix3N::from_column_slice(&[
             0.000000,  0.000000,  1.000000,
             0.957729,  0.000000,  0.287673,
            -0.478864,  0.829418,  0.287673,
            -0.478864, -0.829418,  0.287673,
             0.389831,  0.675207, -0.626200,
            -0.779662,  0.000000, -0.626200,
             0.389831, -0.675207, -0.626200
        ]),
        rotation_basis: vec![
            make_rotation(&[0, 3, 1, 2, 6, 4, 5]), // C3 axial
        ],
        tetrahedra: vec![
            [Vertex(0), Vertex(1), Vertex(2), Vertex(3)],
            [Vertex(0), Vertex(4), Vertex(5), Vertex(6)],

        ],
        mirror: make_mirror(&[0, 3, 2, 1, 6, 5, 4])
    };

    /// Spherized J49 solid in C2v
    ///
    /// Coordinates from [V(CO)7]+ in C2v, from same source as CappedOctahedron, 
    /// minimized to local minimum in Thomson potential
    pub static ref CAPPEDTRIGONALPRISM: Shape = Shape {
        name: Name::CappedTrigonalPrism,
        coordinates: Matrix3N::from_column_slice(&[
           -0.000000, -0.000000,  1.000000,
            0.984798, -0.069552,  0.159173,
           -0.069552,  0.984798,  0.159173,
           -0.984798,  0.069552,  0.159173,
            0.069552, -0.984798,  0.159173,
            0.413726,  0.413726, -0.810964,
           -0.413726, -0.413726, -0.810964
        ]),
        rotation_basis: vec![
            make_rotation(&[0, 3, 4, 1, 2, 6, 5]), // C2 axial
        ],
        tetrahedra: vec![
            [Vertex(0), Vertex(1), Vertex(2), Vertex(5)],
            [Vertex(0), Vertex(3), Vertex(4), Vertex(6)],

        ],
        mirror: make_mirror(&[0, 2, 1, 4, 3, 5, 6])
    };

    pub static ref SQUAREANTIPRISM: Shape = Shape {
        name: Name::SquareAntiprism,
        coordinates: Matrix3N::from_column_slice(&[
             0.607781,  0.607781,  0.511081,
            -0.607781,  0.607781,  0.511081,
            -0.607781, -0.607781,  0.511081,
             0.607781, -0.607781,  0.511081,
             0.859533,  0.000000, -0.511081,
             0.000000,  0.859533, -0.511081,
            -0.859533,  0.000000, -0.511081,
            -0.000000, -0.859533, -0.511081
        ]),
        rotation_basis: vec![
            make_rotation(&[3, 0, 1, 2, 7, 4, 5, 6]), // C4 axial
            make_rotation(&[5, 4, 7, 6, 1, 0, 3, 2]), // C2'-ish
        ],
        tetrahedra: vec![
            [Vertex(0), Vertex(1), Vertex(4), Vertex(5)],
            [Vertex(1), Vertex(2), Vertex(5), Vertex(6)],
            [Vertex(2), Vertex(3), Vertex(6), Vertex(7)],
            [Vertex(3), Vertex(0), Vertex(7), Vertex(4)],

        ],
        mirror: make_mirror(&[2, 1, 0, 3, 5, 4, 7, 6])
    };
    
    pub static ref CUBE: Shape = Shape {
        name: Name::Cube,
        coordinates: Matrix3N::from_column_slice(&[
              0.577350,  0.577350,  0.577350,
              0.577350, -0.577350,  0.577350,
              0.577350, -0.577350, -0.577350,
              0.577350,  0.577350, -0.577350,
             -0.577350,  0.577350,  0.577350,
             -0.577350, -0.577350,  0.577350,
             -0.577350, -0.577350, -0.577350,
             -0.577350,  0.577350, -0.577350
        ]),
        rotation_basis: vec![
            make_rotation(&[3, 0, 1, 2, 7, 4, 5, 6]), // C4
            make_rotation(&[4, 5, 1, 0, 7, 6, 2, 3]), // C4'
        ],
        tetrahedra: vec![
            [Vertex(0), Vertex(1), Vertex(3), Vertex(5)],
            [Vertex(2), Vertex(4), Vertex(6), Vertex(7)],

        ],
        mirror: make_mirror(&[1, 0, 3, 2, 5, 4, 7, 6])
    };

    /// Snub disphenoid, spherized J84 solid in D2d
    pub static ref TRIGONALDODECAHEDRON: Shape = Shape {
        name: Name::TrigonalDodecahedron,
        coordinates: Matrix3N::from_column_slice(&[
              0.620913,  0.000000, -0.783880,
             -0.620913,  0.000000, -0.783880,
              0.000000,  0.620913,  0.783880,
             -0.000000, -0.620913,  0.783880,
              0.950273,  0.000000,  0.311417,
             -0.950273,  0.000000,  0.311417,
              0.000000,  0.950273, -0.311417,
              0.000000, -0.950273, -0.311417
        ]),
        rotation_basis: vec![
            make_rotation(&[1, 0, 3, 2, 5, 4, 7, 6]), // C2z between 01
            make_rotation(&[2, 3, 0, 1, 6, 7, 4, 5]), // C2x + C4z
        ],
        tetrahedra: vec![
            [Vertex(4), Vertex(2), Vertex(3), Vertex(5)],
            [Vertex(0), Vertex(6), Vertex(7), Vertex(1)],

        ],
        mirror: make_mirror(&[0, 1, 3, 2, 4, 5, 7, 6])
    };

    pub static ref HEXAGONALBIPYRAMID: Shape = Shape {
        name: Name::HexagonalBipyramid,
        coordinates: Matrix3N::from_column_slice(&[
             1.000000,  0.000000,  0.000000,
             0.500000,  0.866025,  0.000000,
            -0.500000,  0.866025,  0.000000,
            -1.000000,  0.000000,  0.000000,
            -0.500000, -0.866025,  0.000000,
             0.500000, -0.866025,  0.000000,
             0.000000,  0.000000,  1.000000,
             0.000000,  0.000000, -1.000000
        ]),
        rotation_basis: vec![
            make_rotation(&[5, 0, 1, 2, 3, 4, 6, 7]), // axial C6
            make_rotation(&[0, 5, 4, 3, 2, 1, 7, 6]), // C2 around 0-3
        ],
        tetrahedra: vec![
            [Vertex(6), Vertex(0), Vertex(1), Vertex(7)],
            [Vertex(6), Vertex(4), Vertex(5), Vertex(7)],
            [Vertex(6), Vertex(2), Vertex(3), Vertex(7)],

        ],
        mirror: make_mirror(&[0, 5, 4, 3, 2, 1, 6, 7])
    };

    /// Spherized J50 solid in C4v
    ///
    /// Square-face tricapped. Coordinates are solution to Thomson problem with
    /// nine particles.
    pub static ref TRICAPPEDTRIGONALPRISM: Shape = Shape {
        name: Name::TricappedTrigonalPrism,
        coordinates: Matrix3N::from_column_slice(&[
             0.914109572223, -0.182781178690, -0.361931942064,
             0.293329304506,  0.734642489361, -0.611766566546,
            -0.480176899428, -0.046026929940,  0.875963279468,
            -0.705684904851,  0.704780196051, -0.072757750931,
             0.370605109670,  0.769162968265,  0.520615194684,
            -0.904030464226, -0.412626217894, -0.111662545460,
            -0.162180419233, -0.247163999394, -0.955304908927,
             0.063327560246, -0.997971078243, -0.006583851785,
             0.610701141906, -0.322016246902,  0.723429092590
        ]),
        rotation_basis: vec![
            make_rotation(&[7, 8, 3, 4, 2, 1, 0, 6, 5]), // C3 ccw between 2-4-3
            make_rotation(&[2, 5, 0, 6, 7, 1, 3, 4, 8]), // C2 at 8
        ],
        tetrahedra: vec![
            [Vertex(1), Vertex(0), Vertex(6), Vertex(7)],
            [Vertex(5), Vertex(2), Vertex(3), Vertex(4)],

        ],
        mirror: make_mirror(&[7, 5, 4, 3, 2, 1, 6, 0, 8])
    };

    /// Spherized J10 solid in C4v, to local minimum of Thomson potential
    pub static ref CAPPEDSQUAREANTIPRISM: Shape = Shape {
        name: Name::CappedSquareAntiprism,
        coordinates: Matrix3N::from_column_slice(&[
             -0.000000,  0.932111,  0.362172,
             -0.000000, -0.932111,  0.362172,
              0.932111, -0.000000,  0.362172,
             -0.932111,  0.000000,  0.362172,
              0.559626,  0.559626, -0.611258,
              0.559626, -0.559626, -0.611258,
             -0.559626,  0.559626, -0.611258,
             -0.559626, -0.559626, -0.611258,
              0.000000,  0.000000,  1.000000
        ]),
        rotation_basis: vec![
            make_rotation(&[2, 3, 1, 0, 5, 7, 4, 6, 8]),
        ],
        tetrahedra: vec![
            [Vertex(6), Vertex(3), Vertex(0), Vertex(4)],
            [Vertex(7), Vertex(5), Vertex(1), Vertex(2)],

        ],
        mirror: make_mirror(&[0, 1, 3, 2, 6, 7, 4, 5, 8])
    };

    pub static ref HEPTAGONALBIPYRAMID: Shape = Shape {
        name: Name::HeptagonalBipyramid,
        coordinates: Matrix3N::from_column_slice(&[
              1.000000,  0.000000,  0.000000,
              0.623490,  0.781831,  0.000000,
             -0.222521,  0.974928,  0.000000,
             -0.900969,  0.433884,  0.000000,
             -0.900969, -0.433884,  0.000000,
             -0.222521, -0.974928,  0.000000,
              0.623490, -0.781831,  0.000000,
              0.000000,  0.000000,  1.000000,
              0.000000,  0.000000, -1.000000
        ]),
        rotation_basis: vec![
            make_rotation(&[6, 0, 1, 2, 3, 4, 5, 7, 8]), // axial C7
            make_rotation(&[0, 6, 5, 4, 3, 2, 1, 8, 7]), // C2 around 1 and between 4 and 5
        ],
        tetrahedra: vec![
            [Vertex(8), Vertex(1), Vertex(0), Vertex(7)],
            [Vertex(8), Vertex(3), Vertex(2), Vertex(7)],
            [Vertex(8), Vertex(5), Vertex(4), Vertex(7)],

        ],
        mirror: make_mirror(&[0, 6, 5, 4, 3, 2, 1, 7, 8])
    };

    /// Bicapped square antiprism shape, spherized J17 shape in D4h
    ///
    /// Coordinates are solution to Thomson problem with 10 particles
    pub static ref BICAPPEDSQUAREANTIPRISM: Shape = Shape {
        name: Name::BicappedSquareAntiprism,
        coordinates: Matrix3N::from_column_slice(&[
             0.978696890330,  0.074682616274,  0.191245663177,
             0.537258145625,  0.448413180814, -0.714338368164,
            -0.227939324473, -0.303819959434, -0.925060590777,
             0.274577116268,  0.833436432027,  0.479573895237,
            -0.599426405232,  0.240685139624,  0.763386303437,
            -0.424664555168,  0.830194107787, -0.361161679833,
            -0.402701180119, -0.893328907767,  0.199487398294,
             0.552788606831, -0.770301636525, -0.317899583084,
             0.290107593166, -0.385278374104,  0.876012647646,
            -0.978696887344, -0.074682599351, -0.191245685067
        ]),
        rotation_basis: vec![
            make_rotation(&[0, 7, 6, 1, 5, 2, 4, 8, 3, 9]), // C4z
            make_rotation(&[9, 5, 3, 2, 7, 1, 8, 4, 6, 0]), // C2x + C8z
        ],
        tetrahedra: vec![
            [Vertex(0), Vertex(6), Vertex(8), Vertex(7)],
            [Vertex(9), Vertex(3), Vertex(4), Vertex(5)],
            [Vertex(9), Vertex(2), Vertex(1), Vertex(5)],

        ],
        mirror: make_mirror(&[0, 1, 5, 7, 6, 2, 4, 3, 8, 9])
    };

    /// Coordinates are solution to Thomson problem with 11 particles
    pub static ref EDGECONTRACTEDICOSAHEDRON: Shape = Shape {
        name: Name::EdgeContractedIcosahedron,
        coordinates: Matrix3N::from_column_slice(&[
             0.153486836562, -0.831354332797,  0.534127105044,
             0.092812115769,  0.691598091278, -0.716294626049,
             0.686120068086,  0.724987503180,  0.060269166267,
             0.101393837471,  0.257848797505,  0.960850293931,
            -0.143059218646, -0.243142754178, -0.959382958495,
            -0.909929380017,  0.200934944687, -0.362841110384,
            -0.405338453688,  0.872713317547,  0.272162090194,
             0.896918545883, -0.184616420020,  0.401813264476,
             0.731466092268, -0.415052523977, -0.541007170195,
            -0.439821168531, -0.864743799130, -0.242436592901,
            -0.773718984882, -0.203685975092,  0.599892453681
        ]),
        rotation_basis: vec![
            make_rotation(&[1, 0, 9, 5, 7, 3, 10, 4, 8, 2, 6]), // C2
        ],
        tetrahedra: vec![
            [Vertex(6), Vertex(1), Vertex(5), Vertex(4)],
            [Vertex(3), Vertex(10), Vertex(0), Vertex(9)],
            [Vertex(1), Vertex(2), Vertex(8), Vertex(7)],

        ],
        mirror: make_mirror(&[2, 9, 0, 3, 4, 5, 10, 7, 8, 1, 6])
    };

    pub static ref ICOSAHEDRON: Shape = Shape {
        name: Name::Icosahedron,
        coordinates: Matrix3N::from_column_slice(&[
             0.525731,  0.000000,  0.850651,
             0.525731,  0.000000, -0.850651,
            -0.525731,  0.000000,  0.850651,
            -0.525731,  0.000000, -0.850651,
             0.850651,  0.525731,  0.000000,
             0.850651, -0.525731,  0.000000,
            -0.850651,  0.525731,  0.000000,
            -0.850651, -0.525731,  0.000000,
             0.000000,  0.850651,  0.525731,
             0.000000,  0.850651, -0.525731,
             0.000000, -0.850651,  0.525731,
             0.000000, -0.850651, -0.525731
        ]),
        rotation_basis: vec![
            make_rotation(&[0, 11, 8, 3, 5, 10, 9, 6, 4, 1, 2, 7]), // C5 around 0-3
            make_rotation(&[8, 5, 6, 11, 4, 0, 3, 7, 9, 1, 2, 10]), // C5 around 4-7
            make_rotation(&[2, 3, 0, 1, 7, 6, 5, 4, 10, 11, 8, 9]),// C2 between 0-2 / 1-3
        ],
        tetrahedra: vec![
            [Vertex(0), Vertex(2), Vertex(10), Vertex(8)],
            [Vertex(1), Vertex(3), Vertex(9), Vertex(11)],
            [Vertex(1), Vertex(4), Vertex(5), Vertex(0)],
            [Vertex(3), Vertex(7), Vertex(6), Vertex(2)]
        ],
        mirror: make_mirror(&[0, 1, 2, 3, 5, 4, 7, 6, 10, 11, 8, 9])
    };

    pub static ref CUBOCTAHEDRON: Shape = Shape {
        name: Name::Cuboctahedron,
        coordinates: Matrix3N::from_column_slice(&[
             0.707107,  0.000000,  0.707107,
             0.707107,  0.000000, -0.707107,
            -0.707107,  0.000000,  0.707107,
            -0.707107,  0.000000, -0.707107,
             0.707107,  0.707107,  0.000000,
             0.707107, -0.707107,  0.000000,
            -0.707107,  0.707107,  0.000000,
            -0.707107, -0.707107,  0.000000,
             0.000000,  0.707107,  0.707107,
             0.000000,  0.707107, -0.707107,
             0.000000, -0.707107,  0.707107,
             0.000000, -0.707107, -0.707107
        ]),
        rotation_basis: vec![
            make_rotation(&[10, 11, 8, 9, 5, 7, 4, 6, 0, 1, 2, 3]), // C4 ccw 0-8-2-10
            make_rotation(&[2, 0, 3, 1, 8, 10, 9, 11, 6, 4, 7, 5]), // C4 ccw 4-9-6-8
            make_rotation(&[7, 6, 5, 4, 3, 2, 1, 0, 11, 9, 10, 8]), // C2 along 9-10
        ],
        tetrahedra: vec![
            [ORIGIN, Vertex(6),  Vertex(9),  Vertex(8)],
            [ORIGIN, Vertex(4),  Vertex(1),  Vertex(0)],
            [ORIGIN, Vertex(5), Vertex(11), Vertex(10)],
            [ORIGIN, Vertex(7),  Vertex(3),  Vertex(2)]
        ],
        mirror: make_mirror(&[8, 9, 10, 11, 4, 6, 5, 7, 0, 1, 2, 3])
    };

    pub static ref SHAPES: Vec<&'static Shape> = vec![&LINE, &BENT, &EQUILATERAL_TRIANGLE, &VACANT_TETRAHEDRON, &TSHAPE, &TETRAHEDRON, &SQUARE, &SEESAW, &TRIGONALPYRAMID, &SQUAREPYRAMID, &TRIGONALBIPYRAMID, &PENTAGON, &OCTAHEDRON, &TRIGONALPRISM, &PENTAGONALPYRAMID, &HEXAGON, &PENTAGONALBIPYRAMID, &CAPPEDOCTAHEDRON, &CAPPEDTRIGONALPRISM, &SQUAREANTIPRISM, &CUBE, &TRIGONALDODECAHEDRON, &HEXAGONALBIPYRAMID, &TRICAPPEDTRIGONALPRISM, &CAPPEDSQUAREANTIPRISM, &HEPTAGONALBIPYRAMID, &BICAPPEDSQUAREANTIPRISM, &EDGECONTRACTEDICOSAHEDRON, &ICOSAHEDRON, &CUBOCTAHEDRON];
}

