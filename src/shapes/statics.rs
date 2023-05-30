use crate::shapes::{Shape, Name, Rotation, Matrix3N};
use crate::permutation::Permutation;

// TODO 
// - Improve coordinates and tighten MAX_COLUMN_DEVIATION

use std::f64::consts::{FRAC_1_SQRT_2, SQRT_2};

const SQRT_FRAC_1_3: f64 = 0.5773502691896257;
const SQRT_3: f64 = 1.7320508075688772;
const PENTAGON_X1: f64 = 0.309016994374947;
const PENTAGON_Y1: f64 = 0.951056516295154;
const PENTAGON_X2: f64 = -0.809016994374947;
const PENTAGON_Y2: f64 = 0.587785252292473;
const ICO_1: f64 = 0.5257311121191336;
const ICO_2: f64 = 0.85065080835204;

fn make_rotation(slice: &[usize]) -> Rotation {
    Rotation::new(Permutation {sigma: slice.to_vec()})
}

lazy_static! {
    pub static ref LINE: Shape = Shape {
        name: Name::Line,
        coordinates: Matrix3N::from_column_slice(&[
             1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0
        ]),
        rotation_basis: vec![make_rotation(&[1, 0])],
    };

    /// Bent at 107°
    pub static ref BENT: Shape = Shape {
        name: Name::Bent,
        coordinates: Matrix3N::from_column_slice(&[
                           1.0,               0.0, 0.0,
            -0.292371704722737, 0.956304755963036, 0.0
        ]),
        rotation_basis: vec![make_rotation(&[1, 0])],
    };

    pub static ref EQUILATERAL_TRIANGLE: Shape = Shape {
        name: Name::EquilateralTriangle,
        coordinates: Matrix3N::from_column_slice(&[
             1.0,           0.0, 0.0,
            -0.5,  SQRT_3 / 2.0, 0.0,
            -0.5, -SQRT_3 / 2.0, 0.0
        ]),
        rotation_basis: vec![
            make_rotation(&[1, 2, 0]),
            make_rotation(&[0, 2, 1])
        ],
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
        rotation_basis: vec![make_rotation(&[2, 0, 1])]
    };

    pub static ref TSHAPE: Shape = Shape {
        name: Name::T,
        coordinates: Matrix3N::from_column_slice(&[
            -1.0, -0.0, -0.0,
             0.0,  1.0,  0.0,
             1.0,  0.0,  0.0,
        ]),
        rotation_basis: vec![Rotation::new(Permutation {sigma: vec![2, 1, 0]})],
    };

    pub static ref TETRAHEDRON: Shape = Shape {
        name: Name::Tetrahedron,
        coordinates: Matrix3N::from_column_slice(&[
                 -SQRT_2 / 3.0,  SQRT_2 / SQRT_3, -1.0 / 3.0,
                           0.0,              0.0,        1.0,
            2.0 * SQRT_2 / 3.0,              0.0, -1.0 / 3.0,
                 -SQRT_2 / 3.0, -SQRT_2 / SQRT_3, -1.0 / 3.0
        ]),
        rotation_basis: vec![
            make_rotation(&[0, 3, 1, 2]),
            make_rotation(&[2, 1, 3, 0]),
            make_rotation(&[3, 0, 2, 1]),
            make_rotation(&[1, 2, 0, 3])
        ],
    };

    pub static ref SQUARE: Shape = Shape {
        name: Name::Square,
        coordinates: Matrix3N::from_column_slice(&[
             1.0,  0.0,  0.0,
             0.0,  1.0,  0.0,
            -1.0,  0.0,  0.0,
             0.0, -1.0,  0.0
        ]),
        rotation_basis: vec![
            make_rotation(&[3, 0, 1, 2]),
            make_rotation(&[1, 0, 3, 2]),
            make_rotation(&[3, 2, 1, 0]),
        ],
    };

    /// Equatorially monovacant trigonal bipyramid or edge-centered tetragonal disphenoid
    pub static ref SEESAW: Shape = Shape {
        name: Name::Seesaw,
        coordinates: Matrix3N::from_column_slice(&[
             0.0,  1.0,           0.0,
             1.0,  0.0,           0.0,
            -0.5,  0.0, -SQRT_3 / 2.0,
             0.0, -1.0,           0.0
        ]),
        rotation_basis: vec![make_rotation(&[3, 2, 1, 0])],
    };

    /// Face-centered trigonal pyramid = trig. pl. + axial ligand 
    /// (or monovacant trigonal bipyramid)
    pub static ref TRIGONALPYRAMID: Shape = Shape {
        name: Name::TrigonalPyramid,
        coordinates: Matrix3N::from_column_slice(&[
             1.0,           0.0, 0.0,
            -0.5,  SQRT_3 / 2.0, 0.0,
            -0.5, -SQRT_3 / 2.0, 0.0,
             0.0,           0.0, 1.0
        ]),
        rotation_basis: vec![make_rotation(&[2, 0, 1, 3])],
    };

    /// J1 solid (central position is square-face centered)
    pub static ref SQUAREPYRAMID: Shape = Shape {
        name: Name::SquarePyramid,
        coordinates: Matrix3N::from_column_slice(&[
             1.0,  0.0, 0.0,
             0.0,  1.0, 0.0,
            -1.0,  0.0, 0.0,
             0.0, -1.0, 0.0,
             0.0,  0.0, 1.0,
        ]),
        rotation_basis: vec![make_rotation(&[3, 0, 1, 2, 4])],
    };

    /// J12 solid
    pub static ref TRIGONALBIPYRAMID: Shape = Shape {
        name: Name::TrigonalBipyramid,
        coordinates: Matrix3N::from_column_slice(&[
             1.0,           0.0, 0.0,
            -0.5,  SQRT_3 / 2.0, 0.0,
            -0.5, -SQRT_3 / 2.0, 0.0,
             0.0,           0.0, 1.0,
             0.0,           0.0, -1.0
        ]),
        rotation_basis: vec![
            make_rotation(&[2, 0, 1, 3, 4]), // C3
            make_rotation(&[0, 2, 1, 4, 3]), // C2 on 0
        ],
    };

    pub static ref PENTAGON: Shape = Shape {
        name: Name::Pentagon,
        coordinates: Matrix3N::from_column_slice(&[
                    1.0,          0.0, 0.0,
            PENTAGON_X1,  PENTAGON_Y1, 0.0,
            PENTAGON_X2,  PENTAGON_Y2, 0.0,
            PENTAGON_X2, -PENTAGON_Y2, 0.0,
            PENTAGON_X1, -PENTAGON_Y1, 0.0
        ]),
        rotation_basis: vec![
            make_rotation(&[4, 0, 1, 2, 3]),
            make_rotation(&[0, 4, 3, 2, 1]),
        ],
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
    };
    
    /// J2 solid
    pub static ref PENTAGONALPYRAMID: Shape = Shape {
        name: Name::PentagonalPyramid,
        coordinates: Matrix3N::from_column_slice(&[
                    1.0,          0.0, 0.0,
            PENTAGON_X1,  PENTAGON_Y1, 0.0,
            PENTAGON_X2,  PENTAGON_Y2, 0.0,
            PENTAGON_X2, -PENTAGON_Y2, 0.0,
            PENTAGON_X1, -PENTAGON_Y1, 0.0,
                    0.0,          0.0, 1.0
        ]),
        rotation_basis: vec![
            make_rotation(&[4, 0, 1, 2, 3, 5]),
        ],
    };

    pub static ref HEXAGON: Shape = Shape {
        name: Name::Hexagon,
        coordinates: Matrix3N::from_column_slice(&[
             1.0,           0.0,  0.0,
             0.5,  SQRT_3 / 2.0,  0.0,
            -0.5,  SQRT_3 / 2.0,  0.0,
            -1.0,           0.0,  0.0,
            -0.5, -SQRT_3 / 2.0,  0.0,
             0.5, -SQRT_3 / 2.0,  0.0
        ]),
        rotation_basis: vec![
            make_rotation(&[5, 0, 1, 2, 3, 4]),
            make_rotation(&[0, 5, 4, 3, 2, 1]),
        ],
    };

    /// J13 solid
    pub static ref PENTAGONALBIPYRAMID: Shape = Shape {
        name: Name::PentagonalBipyramid,
        coordinates: Matrix3N::from_column_slice(&[
                    1.0,          0.0,  0.0,
            PENTAGON_X1,  PENTAGON_Y1,  0.0,
            PENTAGON_X2,  PENTAGON_Y2,  0.0,
            PENTAGON_X2, -PENTAGON_Y2,  0.0,
            PENTAGON_X1, -PENTAGON_Y1,  0.0,
                    0.0,          0.0,  1.0,
                    0.0,          0.0, -1.0
        ]),
        rotation_basis: vec![
            make_rotation(&[4, 0, 1, 2, 3, 5, 6]),
            make_rotation(&[1, 0, 4, 3, 2, 6, 5]),
        ],
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
    };
    
    pub static ref CUBE: Shape = Shape {
        name: Name::Cube,
        coordinates: Matrix3N::from_column_slice(&[
              SQRT_FRAC_1_3,  SQRT_FRAC_1_3,  SQRT_FRAC_1_3,
              SQRT_FRAC_1_3, -SQRT_FRAC_1_3,  SQRT_FRAC_1_3,
              SQRT_FRAC_1_3, -SQRT_FRAC_1_3, -SQRT_FRAC_1_3,
              SQRT_FRAC_1_3,  SQRT_FRAC_1_3, -SQRT_FRAC_1_3,
             -SQRT_FRAC_1_3,  SQRT_FRAC_1_3,  SQRT_FRAC_1_3,
             -SQRT_FRAC_1_3, -SQRT_FRAC_1_3,  SQRT_FRAC_1_3,
             -SQRT_FRAC_1_3, -SQRT_FRAC_1_3, -SQRT_FRAC_1_3,
             -SQRT_FRAC_1_3,  SQRT_FRAC_1_3, -SQRT_FRAC_1_3
        ]),
        rotation_basis: vec![
            make_rotation(&[3, 0, 1, 2, 7, 4, 5, 6]), // C4
            make_rotation(&[4, 5, 1, 0, 7, 6, 2, 3]), // C4'
        ],
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
    };

    pub static ref HEXAGONALBIPYRAMID: Shape = Shape {
        name: Name::HexagonalBipyramid,
        coordinates: Matrix3N::from_column_slice(&[
             1.0,           0.0,  0.0,
             0.5,  SQRT_3 / 2.0,  0.0,
            -0.5,  SQRT_3 / 2.0,  0.0,
            -1.0,           0.0,  0.0,
            -0.5, -SQRT_3 / 2.0,  0.0,
             0.5, -SQRT_3 / 2.0,  0.0,
             0.0,           0.0,  1.0,
             0.0,           0.0, -1.0
        ]),
        rotation_basis: vec![
            make_rotation(&[5, 0, 1, 2, 3, 4, 6, 7]), // axial C6
            make_rotation(&[0, 5, 4, 3, 2, 1, 7, 6]), // C2 around 0-3
        ],
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
    };

    pub static ref ICOSAHEDRON: Shape = Shape {
        name: Name::Icosahedron,
        coordinates: Matrix3N::from_column_slice(&[
             ICO_1,    0.0,  ICO_2,
             ICO_1,    0.0, -ICO_2,
            -ICO_1,    0.0,  ICO_2,
            -ICO_1,    0.0, -ICO_2,
             ICO_2,  ICO_1,    0.0,
             ICO_2, -ICO_1,    0.0,
            -ICO_2,  ICO_1,    0.0,
            -ICO_2, -ICO_1,    0.0,
               0.0,  ICO_2,  ICO_1,
               0.0,  ICO_2, -ICO_1,
               0.0, -ICO_2,  ICO_1,
               0.0, -ICO_2, -ICO_1
        ]),
        rotation_basis: vec![
            make_rotation(&[0, 11, 8, 3, 5, 10, 9, 6, 4, 1, 2, 7]), // C5 around 0-3
            make_rotation(&[8, 5, 6, 11, 4, 0, 3, 7, 9, 1, 2, 10]), // C5 around 4-7
            make_rotation(&[2, 3, 0, 1, 7, 6, 5, 4, 10, 11, 8, 9]),// C2 between 0-2 / 1-3
        ],
    };

    pub static ref CUBOCTAHEDRON: Shape = Shape {
        name: Name::Cuboctahedron,
        coordinates: Matrix3N::from_column_slice(&[
             FRAC_1_SQRT_2,            0.0,  FRAC_1_SQRT_2,
             FRAC_1_SQRT_2,            0.0, -FRAC_1_SQRT_2,
            -FRAC_1_SQRT_2,            0.0,  FRAC_1_SQRT_2,
            -FRAC_1_SQRT_2,            0.0, -FRAC_1_SQRT_2,
             FRAC_1_SQRT_2,  FRAC_1_SQRT_2,            0.0,
             FRAC_1_SQRT_2, -FRAC_1_SQRT_2,            0.0,
            -FRAC_1_SQRT_2,  FRAC_1_SQRT_2,            0.0,
            -FRAC_1_SQRT_2, -FRAC_1_SQRT_2,            0.0,
                       0.0,  FRAC_1_SQRT_2,  FRAC_1_SQRT_2,
                       0.0,  FRAC_1_SQRT_2, -FRAC_1_SQRT_2,
                       0.0, -FRAC_1_SQRT_2,  FRAC_1_SQRT_2,
                       0.0, -FRAC_1_SQRT_2, -FRAC_1_SQRT_2
        ]),
        rotation_basis: vec![
            make_rotation(&[10, 11, 8, 9, 5, 7, 4, 6, 0, 1, 2, 3]), // C4 ccw 0-8-2-10
            make_rotation(&[2, 0, 3, 1, 8, 10, 9, 11, 6, 4, 7, 5]), // C4 ccw 4-9-6-8
            make_rotation(&[7, 6, 5, 4, 3, 2, 1, 0, 11, 9, 10, 8]), // C2 along 9-10
        ],
    };

    /// Coordinates are solution to Thomson problem with 13 particles,
    /// from https://www.mathpages.com/home/kmath005/kmath005.htm
    // pub static ref THIRTEEN: Shape = ShapeC::try_new(
    //     Name::Thirteen,
    //     Matrix3N::from_column_slice(&[
    //          0.05292425965854, -0.83790881423107, -0.54323829188807,
    //          0.47276350840958,  0.34260952805535,  0.81185797797518,
    //         -0.07135658514504,  0.93119822473295,  0.35746063281239,
    //          0.08831940925803, -0.44700081209818,  0.89016288168620,
    //         -0.75587432031132,  0.61343840827196, -0.22879539145416,
    //          0.90782799944293,  0.41932997135609,  0.00327086379254,
    //         -0.36478531288162, -0.88910676226719,  0.27644319632902,
    //         -0.92626033119856, -0.32604057007372, -0.18904852687585,
    //          0.19113378437734,  0.77892379819239, -0.59728183722523,
    //          0.77477170458192, -0.60567004279921,  0.18136318544520,
    //         -0.34356905502620, -0.00742104646959, -0.93909809524762,
    //          0.65967130965493, -0.12653736664967, -0.74082525473928,
    //         -0.68634538101074,  0.15812819553012,  0.70987709622413
    //     ]),
    // ).unwrap();
    //
    // pub static ref BICAPPEDHEXAGONALANTIPRISM: Shape = Shape::try_new(
    //     Name::BicappedHexagonalAntiprism,
    //     Matrix3N::from_column_slice(&[
    //         -0.42027653889173, -0.73294377291920, -0.53494023647147,
    //         -0.50070830812184,  0.86341968659613,  0.06162495416629,
    //          0.19126621672024, -0.47768209343689,  0.85745965033422,
    //         -0.45701676134932, -0.81745234619531,  0.35058143354372,
    //          0.91302963719075,  0.03110530949505, -0.40670547246446,
    //          0.26474666022720, -0.30866494517054, -0.91358368939108,
    //          0.50070830812184, -0.86341968659613, -0.06162495416629,
    //         -0.64471674338173,  0.09942497610529,  0.75792809350869,
    //          0.87628941583669, -0.05340326568397,  0.47881619741491,
    //         -0.99779111814822, -0.04795042333682, -0.04597435639945,
    //          0.12506793692344,  0.54054891782736,  0.83196446954746,
    //         -0.58108081276487,  0.24579811887291, -0.77584043062672,
    //          0.54177824120079,  0.83429746194425,  0.10209839545631,
    //          0.18870386643677,  0.68692206249789, -0.70180405445214
    //     ]),
    // ).unwrap();
    //
    // pub static ref FIFTEEN: Shape = Shape::try_new(
    //     Name::Fifteen,
    //     Matrix3N::from_column_slice(&[
    //          0.30884515058659,  0.90510471121471,  0.29223301438421,
    //         -0.61383434989807,  0.72422572535659,  0.31417270667092,
    //          0.63477771492650, -0.04898298339339,  0.77114066159850,
    //         -0.29735090362312, -0.14633261355766, -0.94348778811593,
    //         -0.09410711734564,  0.31505831863336,  0.94439510075223,
    //          0.62160124792603, -0.21150788488163, -0.75423889001410,
    //         -0.75741390123109, -0.18650603918766,  0.62573131579651,
    //         -0.40186767615877,  0.73155149465663, -0.55075836945470,
    //         -0.64700942364219, -0.75847143186698, -0.07810181022134,
    //         -0.95328547093427,  0.04827912785151, -0.29818775414077,
    //          0.42276756545429,  0.64811731361761, -0.63341260911743,
    //          0.08543893689411, -0.88727271212461, -0.45326297265654,
    //          0.74459248211787, -0.65987690652190,  0.10071992758454,
    //          0.95833999151096,  0.28538597927538,  0.01180269053264,
    //         -0.01149424657101, -0.75877209911729,  0.65125477648682
    //     ]),
    // ).unwrap();
    //
    // pub static ref TRIANGULARFACESIXTEEN: Shape = Shape::try_new(
    //     Name::TriangularFaceSixteen,
    //     Matrix3N::from_column_slice(&[
    //          0.61026321033395,  0.33480593525292, -0.71797200490849,
    //         -0.30745797233437, -0.86445060249846, -0.39773703761164,
    //          0.13519452447863,  0.96060560805202, -0.24281537498687,
    //         -0.06496508402285, -0.23406462029950, -0.97004808714927,
    //         -0.86368521730477,  0.07569250868284,  0.49831565251205,
    //         -0.54649919936684, -0.74316205772186,  0.38606836318716,
    //          0.63126441254231, -0.06531961737248,  0.77281213049739,
    //          0.98880060742958, -0.14553013925446, -0.03308076957205,
    //         -0.40290229799000,  0.49522153591843, -0.76969173611061,
    //         -0.16339703052807, -0.19583729824622,  0.96692769276210,
    //         -0.06858361396705,  0.68493018941426,  0.72537364407881,
    //          0.54926790083381, -0.66759720285613, -0.50261192569645,
    //         -0.67438329095859,  0.73743483342630,  0.03737704275866,
    //          0.36535987282490, -0.85058647444670,  0.37817299324212,
    //          0.71876264722102,  0.64700705216914,  0.25448404940828,
    //         -0.90703946919163, -0.16914965022007, -0.38557463241114
    //     ]),
    // ).unwrap();
    //
    // pub static ref OPPOSINGSQUARESSIXTEEN: Shape = Shape::try_new(
    //     Name::OpposingSquaresSixteen,
    //     Matrix3N::from_column_slice(&[
    //         -0.11564934688362, -0.45611058279735, -0.88237654367376,
    //         -0.21185951021077,  0.97352308270011, -0.08583912501494,
    //          0.63823805374955, -0.76812271983004,  0.05137775809515,
    //          0.64117067366140,  0.71497983188846, -0.27875438513023,
    //         -0.84380809002015,  0.47133611793058,  0.25657390972325,
    //         -0.01268589797089, -0.72495319505908,  0.68868129999754,
    //         -0.77215247737937, -0.40813759573353,  0.48703619436932,
    //          0.32121938264953,  0.75289718550532,  0.57442487434687,
    //          0.14366770349070,  0.41811095882347, -0.89696310798462,
    //         -0.85570268070890, -0.37605299181287, -0.35546739593976,
    //          0.70814947952590, -0.25701684285194, -0.65762197130085,
    //          0.53061198011097, -0.09328858116665,  0.84246552878241,
    //         -0.66159378590018,  0.40013081720988, -0.63418372068133,
    //         -0.18556077245003, -0.96721645976412, -0.17337681416241,
    //          0.98668448540003,  0.06921075410141,  0.14718558960577,
    //         -0.31072919707585,  0.25071022088539,  0.91683790891749
    //     ]),
    // ).unwrap();

    pub static ref SHAPES: Vec<&'static Shape> = vec![&LINE, &BENT, &EQUILATERAL_TRIANGLE, &VACANT_TETRAHEDRON, &TSHAPE, &TETRAHEDRON, &SQUARE, &SEESAW, &TRIGONALPYRAMID, &SQUAREPYRAMID, &TRIGONALBIPYRAMID, &PENTAGON, &OCTAHEDRON, &TRIGONALPRISM, &PENTAGONALPYRAMID, &HEXAGON, &PENTAGONALBIPYRAMID, &CAPPEDOCTAHEDRON, &CAPPEDTRIGONALPRISM, &SQUAREANTIPRISM, &CUBE, &TRIGONALDODECAHEDRON, &HEXAGONALBIPYRAMID, &TRICAPPEDTRIGONALPRISM, &CAPPEDSQUAREANTIPRISM, &HEPTAGONALBIPYRAMID, &BICAPPEDSQUAREANTIPRISM, &EDGECONTRACTEDICOSAHEDRON, &ICOSAHEDRON, &CUBOCTAHEDRON];
}

#[cfg(test)]
mod tests {
    use crate::shapes::*;
    use crate::shapes::similarity::unit_sphere_normalize;
    use crate::strong::matrix::StrongPoints;

    fn make_rotation(slice: &[usize]) -> Rotation {
        Rotation::new(Permutation {sigma: slice.to_vec()})
    }

    #[test]
    fn rotations_are_rotations() {
        for shape in SHAPES.iter() {
            let strong_coords = StrongPoints::new(unit_sphere_normalize(shape.coordinates.clone()));
            // Apply each rotation and quaternion fit without a mapping
            for rotation in shape.rotation_basis.iter() {
                let rotated_coords = strong_coords.apply_bijection(rotation);
                let fit = strong_coords.quaternion_fit_with_rotor(&rotated_coords);
                assert!(fit.msd < 1e-6);
            }
        }
    }

    fn print_total_unit_sphere_deviations() {
        let mut shape_deviations: Vec<(Name, f64)> = SHAPES.iter()
            .map(|s| (s.name, s.coordinates.column_iter()
                      .map(|v| (v.norm() - 1.0).abs())
                      .sum())
            ).collect();
        shape_deviations.sort_by(|(_, a), (_, b)| a.partial_cmp(b).expect("No NaNs"));
        shape_deviations.reverse();

        for (name, deviation) in shape_deviations {
            if deviation > 1e-12 {
                println!("- {}: {:e}", name, deviation);
            }
        }
    }

    const MAX_COLUMN_DEVIATION: f64 = 1e-6;

    #[test]
    fn shape_coordinates_on_unit_sphere() {
        // TODO tighten the threshold here by improving shape coordinates
        for shape in SHAPES.iter() {
            let mut pass = true;
            for (i, col) in shape.coordinates.column_iter().enumerate() {
                let deviation = (col.norm() - 1.0).abs();
                if deviation > MAX_COLUMN_DEVIATION {
                    pass = false;
                    println!("Column {} of {} coordinates are not precisely on the unit sphere, deviation {:e}", i, shape.name, deviation);
                }
            }
            if !pass {
                println!("Total unit sphere deviations: ");
                print_total_unit_sphere_deviations();

                panic!("Shape coordinates not on unit sphere");
            }
        }
    }

    fn annotated_rotation_basis(name: Name) -> Option<Vec<Rotation>> {
        match name {
            Name::Line => Some(vec![make_rotation(&[1, 0])]),
            Name::Bent => Some(vec![make_rotation(&[1, 0])]),
            Name::EquilateralTriangle => Some(vec![
                make_rotation(&[1, 2, 0]),
                make_rotation(&[0, 2, 1])
            ]),
            Name::VacantTetrahedron => Some(vec![make_rotation(&[2, 0, 1])]),
            Name::T => Some(vec![make_rotation(&[2, 1, 0])]),
            Name::Tetrahedron => Some(vec![
                make_rotation(&[0, 3, 1, 2]),
                make_rotation(&[2, 1, 3, 0]),
                make_rotation(&[3, 0, 2, 1]),
                make_rotation(&[1, 2, 0, 3])
            ]),
            Name::Square => Some(vec![
                make_rotation(&[3, 0, 1, 2]),
                make_rotation(&[1, 0, 3, 2]),
                make_rotation(&[3, 2, 1, 0]),
            ]),
            Name::Seesaw => Some(vec![make_rotation(&[3, 2, 1, 0])]),
            Name::TrigonalPyramid => Some(vec![make_rotation(&[2, 0, 1, 3])]),
            Name::SquarePyramid => Some(vec![make_rotation(&[3, 0, 1, 2, 4])]),
            Name::TrigonalBipyramid => Some(vec![
                make_rotation(&[2, 0, 1, 3, 4]), // C3
                make_rotation(&[0, 2, 1, 4, 3]), // C2 on 0
            ]),
            Name::Pentagon => Some(vec![
                make_rotation(&[4, 0, 1, 2, 3]),
                make_rotation(&[0, 4, 3, 2, 1]),
            ]),
            Name::Octahedron => Some(vec![
                make_rotation(&[3, 0, 1, 2, 4, 5]),
                make_rotation(&[0, 5, 2, 4, 1, 3]),
                make_rotation(&[4, 1, 5, 3, 2, 0]), // TODO maybe unnecessary?
            ]),
            Name::TrigonalPrism => Some(vec![
                make_rotation(&[2, 0, 1, 5, 3, 4]), // C3 axial
                make_rotation(&[3, 5, 4, 0, 2, 1]), // C2 between 0, 3
            ]),
            Name::PentagonalPyramid => Some(vec![make_rotation(&[4, 0, 1, 2, 3, 5])]),
            Name::Hexagon => Some(vec![
                make_rotation(&[5, 0, 1, 2, 3, 4]),
                make_rotation(&[0, 5, 4, 3, 2, 1]),
            ]),
            Name::PentagonalBipyramid => Some(vec![
                make_rotation(&[4, 0, 1, 2, 3, 5, 6]),
                make_rotation(&[1, 0, 4, 3, 2, 6, 5]),
            ]),
            Name::CappedOctahedron => Some(vec![
                make_rotation(&[0, 3, 1, 2, 6, 4, 5]), // C3 axial
            ]),
            Name::CappedTrigonalPrism => Some(vec![
                make_rotation(&[0, 3, 4, 1, 2, 6, 5]), // C2 axial
            ]),
            Name::SquareAntiprism => Some(vec![
                make_rotation(&[3, 0, 1, 2, 7, 4, 5, 6]), // C4 axial
                make_rotation(&[5, 4, 7, 6, 1, 0, 3, 2]), // C2'-ish
            ]),
            Name::Cube => Some(vec![
                make_rotation(&[3, 0, 1, 2, 7, 4, 5, 6]), // C4
                make_rotation(&[4, 5, 1, 0, 7, 6, 2, 3]), // C4'
            ]),
            Name::TrigonalDodecahedron => Some(vec![
                make_rotation(&[1, 0, 3, 2, 5, 4, 7, 6]), // C2z between 01
                make_rotation(&[2, 3, 0, 1, 6, 7, 4, 5]), // C2x + C4z
            ]),
            Name::HexagonalBipyramid => Some(vec![
                make_rotation(&[5, 0, 1, 2, 3, 4, 6, 7]), // axial C6
                make_rotation(&[0, 5, 4, 3, 2, 1, 7, 6]), // C2 around 0-3
            ]),
            Name::TricappedTrigonalPrism => Some(vec![
                make_rotation(&[7, 8, 3, 4, 2, 1, 0, 6, 5]), // C3 ccw between 2-4-3
                make_rotation(&[2, 5, 0, 6, 7, 1, 3, 4, 8]), // C2 at 8
            ]),
            Name::CappedSquareAntiprism => Some(vec![
                make_rotation(&[2, 3, 1, 0, 5, 7, 4, 6, 8]),
            ]),
            Name::HeptagonalBipyramid => Some(vec![
                make_rotation(&[6, 0, 1, 2, 3, 4, 5, 7, 8]), // axial C7
                make_rotation(&[0, 6, 5, 4, 3, 2, 1, 8, 7]), // C2 around 1 and between 4 and 5
            ]),
            Name::BicappedSquareAntiprism => Some(vec![
                make_rotation(&[0, 7, 6, 1, 5, 2, 4, 8, 3, 9]), // C4z
                make_rotation(&[9, 5, 3, 2, 7, 1, 8, 4, 6, 0]), // C2x + C8z
            ]),
            Name::EdgeContractedIcosahedron => Some(vec![
                make_rotation(&[1, 0, 9, 5, 7, 3, 10, 4, 8, 2, 6]), // C2
            ]),
            Name::Icosahedron => Some(vec![
                make_rotation(&[0, 11, 8, 3, 5, 10, 9, 6, 4, 1, 2, 7]), // C5 around 0-3
                make_rotation(&[8, 5, 6, 11, 4, 0, 3, 7, 9, 1, 2, 10]), // C5 around 4-7
                make_rotation(&[2, 3, 0, 1, 7, 6, 5, 4, 10, 11, 8, 9]),// C2 between 0-2 / 1-3
            ]),
            Name::Cuboctahedron => Some(vec![
                make_rotation(&[10, 11, 8, 9, 5, 7, 4, 6, 0, 1, 2, 3]), // C4 ccw 0-8-2-10
                make_rotation(&[2, 0, 3, 1, 8, 10, 9, 11, 6, 4, 7, 5]), // C4 ccw 4-9-6-8
                make_rotation(&[7, 6, 5, 4, 3, 2, 1, 0, 11, 9, 10, 8]), // C2 along 9-10
            ]),
        }
    }

    #[test]
    fn find_rotations() {
        for shape in SHAPES.iter() {
            if let Some(annotated_basis) = annotated_rotation_basis(shape.name) {
                let expanded_annotation = Shape::expand_rotation_basis(annotated_basis.as_slice());
                let expanded_basis = shape.generate_rotations();
                assert_eq!(expanded_annotation, expanded_basis);
            }
        }
    }

    fn make_mirror(slice: &[usize]) -> Option<Mirror> {
        Some(make_rotation(slice))
    }

    fn annotated_mirror(name: Name) -> Option<Option<Mirror>> {
        match name {
            Name::Line => Some(None),
            Name::Bent => Some(None),
            Name::EquilateralTriangle => Some(None),
            Name::VacantTetrahedron => Some(make_mirror(&[0, 2, 1])),
            Name::T => Some(None),
            Name::Tetrahedron => Some(make_mirror(&[0, 2, 1, 3])),
            Name::Square => Some(None),
            Name::Seesaw => Some(make_mirror(&[0, 2, 1, 3])),
            Name::TrigonalPyramid => Some(make_mirror(&[0, 2, 1, 3])),
            Name::SquarePyramid => Some(make_mirror(&[1, 0, 3, 2, 4])),
            Name::TrigonalBipyramid => Some(make_mirror(&[0, 2, 1, 3, 4])),
            Name::Pentagon => Some(None),
            Name::Octahedron => Some(make_mirror(&[1, 0, 3, 2, 4, 5])),
            Name::TrigonalPrism => Some(make_mirror(&[0, 2, 1, 3, 5, 4])),
            Name::PentagonalPyramid => Some(make_mirror(&[0, 4, 3, 2, 1, 5])),
            Name::Hexagon => Some(None),
            Name::PentagonalBipyramid => Some(make_mirror(&[0, 4, 3, 2, 1, 5, 6])),
            Name::CappedOctahedron => Some(make_mirror(&[0, 1, 3, 2, 6, 5, 4])),
            Name::CappedTrigonalPrism => Some(make_mirror(&[0, 2, 1, 4, 3, 5, 6])),
            Name::SquareAntiprism => Some(make_mirror(&[0, 3, 2, 1, 5, 4, 7, 6])),
            Name::Cube => Some(make_mirror(&[1, 0, 3, 2, 5, 4, 7, 6])),
            Name::TrigonalDodecahedron => Some(make_mirror(&[0, 1, 3, 2, 4, 5, 7, 6])),
            Name::HexagonalBipyramid => Some(make_mirror(&[0, 5, 4, 3, 2, 1, 6, 7])),
            Name::TricappedTrigonalPrism => Some(make_mirror(&[7, 5, 4, 3, 2, 1, 6, 0, 8])),
            Name::CappedSquareAntiprism => Some(make_mirror(&[0, 1, 3, 2, 6, 7, 4, 5, 8])),
            Name::HeptagonalBipyramid => Some(make_mirror(&[0, 6, 5, 4, 3, 2, 1, 7, 8])),
            Name::BicappedSquareAntiprism => Some(make_mirror(&[0, 1, 5, 7, 6, 2, 4, 3, 8, 9])),
            Name::EdgeContractedIcosahedron => Some(make_mirror(&[2, 9, 0, 3, 4, 5, 10, 7, 8, 1, 6])),
            Name::Icosahedron => Some(make_mirror(&[0, 1, 2, 3, 5, 4, 7, 6, 10, 11, 8, 9])),
            Name::Cuboctahedron => Some(make_mirror(&[8, 9, 10, 11, 4, 6, 5, 7, 0, 1, 2, 3])),
        }
    }

    #[test]
    fn find_mirror() {
        // TODO temporary test size limit
        for shape in SHAPES.iter().filter(|shape| shape.num_vertices() <= 8) {
            if let Some(annotated_mirror_maybe) = annotated_mirror(shape.name) {
                let maybe_found_mirror = shape.find_mirror();
                // Either both no mirror, or both Somes
                assert!(annotated_mirror_maybe.as_ref().xor(maybe_found_mirror.as_ref()).is_none());

                if let Some((annotated_mirror, found_mirror)) = annotated_mirror_maybe.zip(maybe_found_mirror) {
                    let rotations = shape.generate_rotations();
                    assert!(Shape::is_rotation(&annotated_mirror, &found_mirror, &rotations));

                    // Mirrors are not rotations
                    assert!(!rotations.contains(&found_mirror));
                }
            }
        }
    }

    fn annotated_tetrahedra(name: Name) -> Option<Vec<[Particle; 4]>> {
        let maybe_vec = match name {
            Name::Line => Some(vec![]),
            Name::Bent => Some(vec![]),
            Name::EquilateralTriangle => Some(vec![]),
            Name::VacantTetrahedron => Some(vec![[None, Some(0), Some(1), Some(2)]]),
            Name::T => Some(vec![]),
            Name::Tetrahedron => Some(vec![[Some(0), Some(1), Some(2), Some(3)]]),
            Name::Square => Some(vec![]),
            Name::Seesaw => Some(vec![
                [Some(0), None, Some(1), Some(2)],
                [None, Some(3), Some(1), Some(2)]
            ]),
            Name::TrigonalPyramid => Some(vec![[Some(0), Some(1), Some(3), Some(2)]]),
            Name::SquarePyramid => Some(vec![
                [Some(0), Some(1), Some(4), None],
                [Some(1), Some(2), Some(4), None],
                [Some(2), Some(3), Some(4), None],
                [Some(3), Some(0), Some(4), None],
            ]),
            Name::TrigonalBipyramid => Some(vec![
                [Some(0), Some(1), Some(3), Some(2)], 
                [Some(0), Some(1), Some(2), Some(4)]
            ]),
            Name::Pentagon => Some(vec![]),
            Name::Octahedron => Some(vec![
                [Some(3), Some(0), Some(4), None],
                [Some(0), Some(1), Some(4), None],
                [Some(1), Some(2), Some(4), None],
                [Some(2), Some(3), Some(4), None],
                [Some(3), Some(0), None, Some(5)],
                [Some(0), Some(1), None, Some(5)],
                [Some(1), Some(2), None, Some(5)],
                [Some(2), Some(3), None, Some(5)],
            ]),
            Name::TrigonalPrism => Some(vec![
                [None, Some(0), Some(2), Some(1)],
                [Some(3), None, Some(5), Some(4)]
            ]),
            Name::PentagonalPyramid => Some(vec![
                [Some(0), None, Some(1), Some(5)],
                [Some(1), None, Some(2), Some(5)],
                [Some(2), None, Some(3), Some(5)],
                [Some(3), None, Some(4), Some(5)],
                [Some(4), None, Some(0), Some(5)],
            ]),
            Name::Hexagon => Some(vec![]),
            Name::PentagonalBipyramid =>Some(vec![
                [Some(0), Some(1), Some(5), Some(6)],
                [Some(1), Some(2), Some(5), Some(6)],
                [Some(2), Some(3), Some(5), Some(6)],
                [Some(3), Some(4), Some(5), Some(6)],
                [Some(4), Some(0), Some(5), Some(6)],
            ]),
            Name::CappedOctahedron => Some(vec![
                [Some(0), Some(1), Some(2), Some(3)],
                [Some(0), Some(4), Some(5), Some(6)],
            ]),
            Name::CappedTrigonalPrism => Some(vec![
                [Some(0), Some(1), Some(2), Some(5)],
                [Some(0), Some(3), Some(4), Some(6)],
            ]),
            Name::SquareAntiprism => Some(vec![
                [Some(0), Some(1), Some(4), Some(5)],
                [Some(1), Some(2), Some(5), Some(6)],
                [Some(2), Some(3), Some(6), Some(7)],
                [Some(3), Some(0), Some(7), Some(4)],
            ]),
            Name::Cube => Some(vec![
                [Some(0), Some(1), Some(3), Some(5)],
                [Some(2), Some(4), Some(6), Some(7)],
            ]),
            Name::TrigonalDodecahedron => Some(vec![
                [Some(4), Some(2), Some(3), Some(5)],
                [Some(0), Some(6), Some(7), Some(1)],
            ]),
            Name::HexagonalBipyramid => Some(vec![
                [Some(6), Some(0), Some(1), Some(7)],
                [Some(6), Some(4), Some(5), Some(7)],
                [Some(6), Some(2), Some(3), Some(7)],
            ]),
            Name::TricappedTrigonalPrism => Some(vec![
                [Some(1), Some(0), Some(6), Some(7)],
                [Some(5), Some(2), Some(3), Some(4)],
            ]),
            Name::CappedSquareAntiprism => Some(vec![
                [Some(6), Some(3), Some(0), Some(4)],
                [Some(7), Some(5), Some(1), Some(2)],
            ]),
            Name::HeptagonalBipyramid => Some(vec![
                [Some(8), Some(1), Some(0), Some(7)],
                [Some(8), Some(3), Some(2), Some(7)],
                [Some(8), Some(5), Some(4), Some(7)],
            ]),
            Name::BicappedSquareAntiprism => Some(vec![
                [Some(0), Some(6), Some(8), Some(7)],
                [Some(9), Some(3), Some(4), Some(5)],
                [Some(9), Some(2), Some(1), Some(5)],
            ]),
            Name::EdgeContractedIcosahedron => Some(vec![
                [Some(6), Some(1), Some(5), Some(4)],
                [Some(3), Some(10), Some(0), Some(9)],
                [Some(1), Some(2), Some(8), Some(7)],
            ]),
            Name::Icosahedron => Some(vec![
                [Some(0), Some(2), Some(10), Some(8)],
                [Some(1), Some(3), Some(9), Some(11)],
                [Some(1), Some(4), Some(5), Some(0)],
                [Some(3), Some(7), Some(6), Some(2)]
            ]),
            Name::Cuboctahedron => Some(vec![
                [None, Some(6),  Some(9),  Some(8)],
                [None, Some(4),  Some(1),  Some(0)],
                [None, Some(5), Some(11), Some(10)],
                [None, Some(7),  Some(3),  Some(2)]
            ]),
        };

        // Transform Option<i32>s into Particles
        maybe_vec.map(|vec| vec.into_iter().map(|arr| arr.map(|o| o.map(Vertex::from)).map(Particle::from)).collect())
    }

    fn tetrahedron_volume(tetrahedron: &[Particle; 4], points: &Matrix3N) -> f64 {
        let zero = Matrix3N::zeros(1);
        let r = |p: Particle| {
            match p {
                Particle::Vertex(v) => points.column(v.get()),
                Particle::Origin => zero.column(0)
            }
        };

        crate::geometry::signed_tetrahedron_volume_with_array(tetrahedron.map(r))
    }


    // NOTE annotated tetrahedra aren't really used at all - can't compare with
    // them really since no idea if they're any better, that'll have to wait
    // until visualization is available
    const MIN_TETRAHEDRON_VOLUME: f64 = 0.4 / 6.0;

    #[test]
    fn annotated_tetrahedra_positive_volume() {
        for shape in SHAPES.iter() {
            if let Some(annotated_tetrahedra) = annotated_tetrahedra(shape.name) {
                let significant_positive_volume = annotated_tetrahedra.iter().all(|tetrahedron| {
                    let volume = tetrahedron_volume(tetrahedron, &shape.coordinates);

                    if volume < MIN_TETRAHEDRON_VOLUME {
                        println!("Shape {} tetrahedron {:?} does not have significant positive volume at V = {:e}", shape.name, tetrahedron, volume);
                    }

                    volume > MIN_TETRAHEDRON_VOLUME
                });
                assert!(significant_positive_volume);
            }
        }
    }
}
