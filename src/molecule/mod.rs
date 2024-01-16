use bitmask::bitmask;
use std::collections::HashMap;

use std::path::Path;

use crate::stereo::Rank;
use crate::strong::IndexBase;
use crate::shapes::{Vertex, Name};
use crate::strong::surjection::Surjection;
use crate::strong::matrix::Positions;
use crate::quaternions::Matrix3N;

use petgraph::unionfind::UnionFind;
use petgraph::visit::{NodeIndexable, IntoEdgeReferences, EdgeRef};

use nuclide::element::Element;
use nuclide::Nuclide;

/// Nucleus is an index that refers to the core of an atom in a molecules
/// and simultaneously a vertex of the molecular graph
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Nucleus(petgraph::graph::NodeIndex);

impl IndexBase for Nucleus {
    type Type = usize;

    fn get(&self) -> Self::Type {
        self.0.index()
    }
}

impl From<usize> for Nucleus {
    fn from(original: usize) -> Nucleus {
        Nucleus(petgraph::graph::NodeIndex::new(original))
    }
}

impl From<Nucleus> for usize {
    fn from(val: Nucleus) -> Self {
        val.0.index()
    }
}

impl From<petgraph::graph::NodeIndex> for Nucleus {
    fn from(val: petgraph::graph::NodeIndex) -> Self {
        Nucleus(val)
    }
}

/// Sites are one or more atoms (graph vertices) bound at a shape vertex
#[derive(IndexBase, Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Site(usize);

#[derive(Clone)]
/// Result of ranking algorithm
pub struct Ranking {
    pub nuclei: Surjection<Nucleus, Rank>,
    pub site: Surjection<Site, Rank>
}

#[derive(Clone)]
pub struct AtomStereo {
    shape: Name,
    ranking: Ranking,
    placement: Surjection<Vertex, Site>,
}
/// Atomic nucleus information
#[derive(Clone)]
pub enum NucleusType {
    /// An natural abundance mix of isotopes
    Element(Element),
    /// A particular isotope marker at this position
    Isotope(Nuclide)
}

/// Bond type information
#[derive(Clone)]
pub enum Bond {
    /// Integer bond order (formal orders up to six exist)
    Regular(i8),
    /// Indicates bond is between member of a multi-atom site and the coordination target of the
    /// site
    Eta
}

#[derive(Clone)]
struct VertexData {
    nucleus: NucleusType
}

#[derive(Clone)]
struct EdgeData {
    bond: Bond
}

/// Underlying foreign data representation of the graph part of the molecule: Element type or
/// monoisotopic atoms as vertices, discrete bond types as edges.
type Graph = petgraph::stable_graph::StableGraph::<VertexData, EdgeData, petgraph::Undirected>;

fn connected_components(graph: &Graph) -> HashMap<Nucleus, Component> {
    let mut reduction = HashMap::new();
    let mut next_reduction: usize = 0;

    let mut vertex_sets = UnionFind::new(graph.node_bound());
    for edge in graph.edge_references() {
        let vertices = [edge.source(), edge.target()];
        let reduced = vertices.map(|node| {
            let value = reduction.entry(node).or_insert(next_reduction);
            next_reduction += 1;
            *value
        });
        vertex_sets.union(reduced[0], reduced[1]);
    }

    let inverted_map: HashMap<usize, petgraph::graph::NodeIndex> = HashMap::from_iter(reduction.into_iter().map(|(key, value)| (value, key)));

    // labeling: (reduced node idx) -> component
    vertex_sets.into_labeling().into_iter()
        .enumerate()
        .map(|(i, c)| (Nucleus(inverted_map[&i]), Component(c)))
        .collect()
}

bitmask! {
    /// Bitmask of `Environ`
    pub mask EnvironMask: u8 where 
    /// Components of an vertex's environment
    flags Environ {
        /// Graph connectivity
        Topo = 1 << 0,
        /// Vertex element or isotopes
        Nuclei = 1 << 1,
        /// Bond orders
        BondOrders = 1 << 2,
        /// Local shapes
        Shape = 1 << 3,
        /// Stereo of atoms and bonds
        Stereo = 1 << 4
    }
}

/// Stereoisomerism representation part of Molecule
pub struct Stereo {
    atoms: HashMap<Nucleus, AtomStereo>,
    // bonds: HashMap<Edge, BondStereo>,
    // etc.
}

pub struct Molecule {
    graph: Graph,
    stereo: Stereo,
    regularized: EnvironMask
}

impl Molecule {
    fn new_fit(graph: Graph, positions: Positions<Nucleus>) -> Molecule {
        todo!()
    }
}

/// Component index 
#[derive(IndexBase, Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Component(usize);

pub struct GraphSplit {
    graphs: Vec<Graph>,
    component_map: HashMap<Nucleus, Component>
}

impl GraphSplit {
    fn by_components(parent: Graph) -> GraphSplit {
        let component_map = connected_components(&parent);
        let components = component_map.values().max().map(|c| c.get() + 1).unwrap_or(0);

        // Shortcut if there's only a single component
        if components <= 1 {
            return GraphSplit {
                graphs: vec![parent],
                component_map
            };
        }

        // Split graphs
        let graphs: Vec<_> = (0..components)
            .map(|component| {
                parent.filter_map(
                    |i, n| (component_map[&Nucleus(i)] == Component(component)).then_some(n.clone()),
                    |_, e| Some(e.clone())
                )
            })
            .collect();

        GraphSplit {graphs, component_map}
    }
}

// With StableGraph and filter_map you can make an IndexMap: Node -> Component since the vertex
// index is retained in the filtered graphs

fn convert_frame(frame: chemfiles::Frame) -> Result<(Graph, Positions<Nucleus>), chemfiles::Error> {
    let topology = frame.topology();
    let positions = {
        let frame_iterator = frame.positions().iter()
            .flat_map(|i| i.iter())
            .cloned();
        let matrix = Matrix3N::from_iterator(topology.size(), frame_iterator);
        Positions::wrap(matrix)
    };

    let mut graph = Graph::with_capacity(topology.size(), topology.bonds_count());
    let mut vertices = Vec::new();

    // Add vertices
    for atom in (0..topology.size()).map(|i| topology.atom(i)) {
        let atomic_number = atom.atomic_number();
        if atomic_number == 0 {
            return Err(chemfiles::Error {
                status: chemfiles::Status::FormatError,
                message: "Undefined atomic number in topology".to_owned(),
            });
        }

        let nucleus: NucleusType = NucleusType::Element(Element::from_protons(atomic_number as u8));
        let index = graph.add_node(VertexData {nucleus});
        vertices.push(index);
    }

    // Add edges
    for [i, j] in topology.bonds() {
        let bond = match topology.bond_order(i, j) {
            chemfiles::BondOrder::Unknown => panic!("Unknown bond in topology to load"),
            chemfiles::BondOrder::Single => Bond::Regular(1),
            chemfiles::BondOrder::Double => Bond::Regular(2),
            chemfiles::BondOrder::Triple => Bond::Regular(3),
            chemfiles::BondOrder::Quadruple => Bond::Regular(4),
            chemfiles::BondOrder::Quintuplet => Bond::Regular(5),
            chemfiles::BondOrder::Amide => Bond::Regular(1),
            chemfiles::BondOrder::Aromatic => Bond::Regular(1),
            _ => panic!("Invalid bond order supplied by chemfiles"),
        };
        graph.add_edge(vertices[i], vertices[j], EdgeData {bond});
    }

    Ok((graph, positions))
}

pub struct Parts {
    graphs: Vec<Graph>,
    component_map: HashMap<Nucleus, Component>,
    positions: Option<Positions<Nucleus>>
}

impl Parts {
    fn new(split: GraphSplit, positions: Positions<Nucleus>) -> Parts {
        // Chemfiles docs state empty positions are completely zeroed
        let zero_vectors = positions.matrix.column_iter()
            .filter(|c| c.norm_squared() < 1e-6)
            .count();
        let positions = (zero_vectors < 2).then_some(positions);

        Parts {
            graphs: split.graphs,
            component_map: split.component_map,
            positions
        }
    }
}

// This is basically io::Interpret
pub fn read(path: impl AsRef<Path>) -> Result<Parts, chemfiles::Error> {
    let mut trajectory = chemfiles::Trajectory::open(path, 'r')?;
    let frame = {
        let mut frame = chemfiles::Frame::new();
        trajectory.read(&mut frame)?;
        frame
    };

    let (graph, positions) = convert_frame(frame)?;

    let splat = GraphSplit::by_components(graph);
    Ok(Parts::new(splat, positions))
}

pub struct Molecules {
    molecules: Vec<Molecule>,
    component_map: HashMap<Nucleus, Component>
}

impl Molecules {
    /// If only one molecule found, yields it
    ///
    /// NOTE: If there is only one molecule, the index mapping is trivial
    fn single(mut self) -> Option<Molecule> {
        if self.molecules.len() != 1 {
            return None;
        }

        Some(self.molecules.pop().unwrap())
    }
}
