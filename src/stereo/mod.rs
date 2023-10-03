use std::fmt::Display;
use std::collections::{HashSet, HashMap};
use gcd::Gcd;
use sorted_vec::SortedVec;

use crate::shapes::{Vertex, Shape, Rotation};
use crate::strong::surjection::{Surjection, Surjectable, SurjectionError};
use crate::strong::bijection::{Bijection, Bijectable, bijections};
use crate::strong::IndexBase;
use crate::permutation::PermutationError;

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Debug)]
/// Ordered homogeneous pair
pub struct OrderedPair<T: Ord>(pub T, pub T);

impl<T: Ord> OrderedPair<T> {
    /// Generate from unordered data
    pub fn new(a: T, b: T) -> Self {
        if b < a {
            Self(b, a)
        } else {
            Self(a, b)
        }
    }

    /// Apply a function to both values and create a new ordered pair from the result
    pub fn map<F, B>(&self, f: F) -> OrderedPair<B> where F: Fn(&T) -> B, B: Ord {
        OrderedPair::new(f(&self.0), f(&self.1))
    }
}

impl<T: Display + Ord> std::fmt::Display for OrderedPair<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OrderedPair({}, {})", self.0, self.1)
    }
}

#[derive(IndexBase, PartialEq, Eq, Copy, Clone, PartialOrd, Ord, Hash, Debug)]
/// Index indicating rank
pub struct Rank(usize);

/// An occupation is a placement of rank indices onto shape vertices
pub type Occupation = Surjection<Vertex, Rank>;
/// A link is a path through the graph between two vertices that does not include the centroid
pub type Link = OrderedPair<Vertex>;

/// Simplified model of sterically unique assignment of ligands to a stereocenter
#[derive(Hash, Eq, PartialEq, Clone, PartialOrd, Ord, Debug)]
pub struct Coordination {
    occupation: Occupation,
    links: SortedVec<Link>
}

impl std::fmt::Display for Coordination {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Coordination {{occupation: {:?}, links: {:?}}}", self.occupation.sigma, self.links)
    }
}

struct RotationEnumeratorLink<'a> {
    rotation_iter: std::slice::Iter<'a, Rotation>,
    coordination: Coordination
}

struct RotationEnumerator<'a> {
    shape: &'a Shape,
    chain: Vec<RotationEnumeratorLink<'a>>,
    found_rotations: HashSet<Coordination>
}

impl RotationEnumerator<'_> {
    fn new(initial: Coordination, shape: &Shape) -> RotationEnumerator {
        let chain = vec![RotationEnumeratorLink {rotation_iter: shape.rotation_basis.iter(), coordination: initial}];
        RotationEnumerator { shape, chain, found_rotations: HashSet::new() }
    }
}

impl Iterator for RotationEnumerator<'_> {
    type Item = Coordination;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Find a coordination that a new rotation can be applied to
            let (truncate_count, rotation) = self.chain.iter_mut()
                .rev()
                .enumerate()
                .find_map(|(count, link)| link.rotation_iter.next().map(|rotation| (count, rotation)))?;

            // Drop all links in the chain that didn't yield a new rotation
            self.chain.truncate(self.chain.len() - truncate_count);

            // Apply the rotation and see if it's new
            let last_coordination = &self.chain.last()?.coordination;
            let trial = last_coordination.biject(rotation).ok()?;
            let is_new = self.found_rotations.insert(trial.clone());
            if is_new {
                self.chain.push(RotationEnumeratorLink {
                    rotation_iter: self.shape.rotation_basis.iter(),
                    coordination: trial.clone()
                });
                return Some(trial);
            }
        }
    }
}

impl Coordination {
    /// Generate a new coordination from unsorted data
    pub fn new_unsorted(occupation: Occupation, unsorted_links: Vec<Link>) -> Self {
        let links = SortedVec::from_unsorted(unsorted_links);
        Coordination {occupation, links}
    }

    /// Generate the maximally asymmetric unlinked Coordination for this size
    pub fn asymmetric(size: usize) -> Self {
        Coordination {
            occupation: Occupation::identity(size), 
            links: Vec::new().into()
        }
    }

    /// Generates all rotationally superimposable coordinations, including itself
    pub fn rotations(&self, shape: &Shape) -> HashSet<Coordination> {
        HashSet::from_iter(RotationEnumerator::new(self.clone(), shape))
    }

    /// Checks whether this coordination is a rotation of another
    ///
    /// NOTE: This includes comparing `self` against `other`, i.e. identical coordinations are also
    /// rotations of one another (with the identity rotation).
    pub fn is_rotation_of(&self, other: &Coordination, shape: &Shape) -> bool {
        RotationEnumerator::new(self.clone(), shape).any(|rotation| &rotation == other)
    }

    /// Checks whether this coordination is an enantiomer of another
    pub fn is_enantiomer_of(&self, other: &Coordination, shape: &Shape) -> bool {
        shape.find_mirror()
            .and_then(|mirror| other.biject(&mirror).ok())
            .map(|mirrored| self.is_rotation_of(&mirrored, shape))
            .unwrap_or(false)
    }

    /// Generate rotationally distinct permutations of this coordination
    ///
    /// For each group of rotationally equivalent coordinations, the lexicographically least is
    /// noted. The returned list of distinct coordinations are also lexicographically sorted.
    ///
    /// NOTE: No considerations regarding feasibility. Links are considered infinitely long and accommodating.
    pub fn rotationally_distinct_permutations(&self, shape: &Shape) -> Vec<(Coordination, usize)> {
        assert!(self.occupation.domain_size() == shape.num_vertices());

        let mut unordered_distinct = Vec::new();
        let mut rotation_forwarding = HashMap::new();

        let base_rotations = self.rotations(shape);
        unordered_distinct.push((base_rotations.iter().min().expect("Matched size").clone(), 1));
        for rotation in base_rotations {
            rotation_forwarding.insert(rotation, 0);
        }

        for bijection in bijections(shape.num_vertices()) {
            let trial = self.biject(&bijection).expect("Shape size matches coordination");
            if let Some(unordered_index) = rotation_forwarding.get(&trial) {
                unordered_distinct[*unordered_index].1 += 1;
            } else {
                // This is a new rotationally distinct coordination
                let trial_rotations = trial.rotations(shape);
                unordered_distinct.push((trial_rotations.iter().min().expect("Matched size").clone(), 1));
                let unordered_index = unordered_distinct.len() - 1;
                for rotation in trial_rotations {
                    rotation_forwarding.insert(rotation, unordered_index);
                }
            }
        }

        let gcd = unordered_distinct.iter().map(|(_, count)| *count).reduce(|a, b| a.gcd(b)).expect("At least one element");
        if gcd != 1 {
            for mut pair in unordered_distinct.iter_mut() {
                pair.1 /= gcd;
            }
        }

        unordered_distinct.sort_unstable_by(|(a, _), (b, _)| a.cmp(b));
        unordered_distinct
    }
}

impl Bijectable<Vertex> for Coordination {
    type T = Vertex;
    type Output = Coordination;

    fn biject(&self, bijection: &Bijection<Vertex, Vertex>) -> Result<Coordination, PermutationError> {
        let occupation = bijection.surject(&self.occupation).map_err(|e| {
            assert_eq!(e, SurjectionError::LengthMismatch);
            PermutationError::LengthMismatch
        })?;
        let links = if self.links.is_empty() {
            SortedVec::new()
        } else {
            let inverse = bijection.inverse();
            let unordered_links: Result<Vec<_>, _> = self.links.iter()
                .map(|OrderedPair(a, b)| -> Result<OrderedPair<Vertex>, PermutationError> {
                    let i = inverse.get(a).ok_or(PermutationError::LengthMismatch)?;
                    let j = inverse.get(b).ok_or(PermutationError::LengthMismatch)?;
                    Ok(OrderedPair::new(i, j))
                })
                .collect();
            SortedVec::from_unsorted(unordered_links?)
        };

        Ok(Coordination {occupation, links})
    }
}


#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use sorted_vec::SortedVec;
    use crate::stereo::{Coordination, OrderedPair, Rank, Link};
    use crate::permutation::Permutation;
    use crate::strong::bijection::{Bijection, Bijectable};
    use crate::strong::surjection::Surjection;
    use crate::shapes::{Vertex, Rotation, Shape, shape_from_name, Name};
    use crate::inertia::angle;

    #[test]
    fn coordination_basics() {
        // Bijection links correctly
        let vertex_pair = |a, b| {OrderedPair::new(Vertex::from(a), Vertex::from(b))};

        let shipscrew = Coordination::new_unsorted(
            Surjection::identity(6),
            vec![vertex_pair(0, 1), vertex_pair(2, 4), vertex_pair(3, 5)]
        );
        let c4z: Rotation = Bijection::new(Permutation::try_from([3, 0, 1, 2, 4, 5]).unwrap());
        let rotated = shipscrew.biject(&c4z).unwrap();

        let expected = Coordination::new_unsorted(
            Bijection::new(Permutation::try_from([3, 0, 1, 2, 4, 5]).unwrap()).into(),
            vec![vertex_pair(0, 5), vertex_pair(1, 2), vertex_pair(3, 4)]
        );

        assert_eq!(rotated, expected);
        // Logic here is that the unique ranks should stay connected as is
       let prior_rank_pairs = shipscrew.links.iter()
            .map(|link| link.map(|vertex| shipscrew.occupation.get(vertex).unwrap()))
            .collect();
        let post_rank_pairs: Vec<_> = expected.links.iter()
            .map(|link| link.map(|vertex| expected.occupation.get(vertex).unwrap()))
            .collect();
        assert_eq!(SortedVec::from_unsorted(prior_rank_pairs), SortedVec::from_unsorted(post_rank_pairs));

        let octahedron = shape_from_name(Name::Octahedron);
        assert_eq!(Coordination::asymmetric(octahedron.num_vertices()).rotations(octahedron).len(), 24);
    }

    fn occupation(slice: &[usize]) -> Surjection<Vertex, Rank> {
        let sigma: Vec<_> = slice.iter().copied().map_into::<Rank>().collect();
        Surjection::try_from(sigma).expect("Valid occupation")
    }

    fn link(a: usize, b: usize) -> Link {
        Link::new(Vertex::from(a), Vertex::from(b))
    }

    fn link_angle(link: &Link, shape: &Shape) -> f64 {
        angle(&shape.coordinates.point(link.0), &shape.coordinates.point(link.1))
    }

    #[test]
    fn coordination_distinct_counts() {
        let tests = vec![
            (Name::Tetrahedron, vec![
                (occupation(&[0, 0, 0, 0]), vec![], 1),
                (occupation(&[0, 0, 0, 1]), vec![], 1),
                (occupation(&[0, 0, 1, 1]), vec![], 1),
                (occupation(&[0, 0, 1, 2]), vec![], 1),
                (occupation(&[0, 1, 2, 3]), vec![], 2),
            ]),
            (Name::Square, vec![
                (occupation(&[0, 0, 0, 0]), vec![], 1),
                (occupation(&[0, 0, 0, 1]), vec![], 1),
                (occupation(&[0, 0, 1, 1]), vec![], 2),
                (occupation(&[0, 0, 1, 2]), vec![], 2),
                (occupation(&[0, 1, 2, 3]), vec![], 3),
            ]),
            (Name::Octahedron, vec![
                (occupation(&[0, 0, 0, 0, 0, 0]), vec![], 1),
                (occupation(&[0, 0, 0, 0, 0, 1]), vec![], 1),
                (occupation(&[0, 0, 0, 0, 1, 1]), vec![], 2),
                (occupation(&[0, 0, 0, 1, 1, 1]), vec![], 2),
                (occupation(&[0, 0, 0, 0, 1, 2]), vec![], 2),
                (occupation(&[0, 0, 0, 1, 1, 2]), vec![], 3),
                (occupation(&[0, 0, 1, 1, 2, 2]), vec![], 6),
                (occupation(&[0, 0, 0, 1, 2, 3]), vec![], 5),
                (occupation(&[0, 0, 1, 1, 2, 3]), vec![], 8),
                (occupation(&[0, 0, 1, 2, 3, 4]), vec![], 15),
                (occupation(&[0, 1, 2, 3, 4, 5]), vec![], 30),
                ( // M(A-A)_3
                    occupation(&[0, 0, 0, 0, 0, 0]), 
                    vec![link(0, 1), link(2, 3), link(4, 5)], 
                    2 // besides the two cis-cis-cis enantiomers, there are cis-cis-trans and
                      // trans-trans-trans!
                ),
                ( // M(A-B)_3
                    occupation(&[0, 1, 0, 1, 0, 1]), 
                    vec![link(0, 1), link(2, 3), link(4, 5)], 
                    4
                ),
                ( // M(A-B)_2CD
                    occupation(&[0, 1, 0, 1, 2, 3]), 
                    vec![link(0, 1), link(2, 3)], 
                    11
                ),
                ( // M(A-A)(B-C)DE
                    occupation(&[0, 0, 1, 2, 3, 4]),
                    vec![link(0, 1), link(2, 3)],
                    10
                ),
                ( // M(A-B)(C-D)EF
                    occupation(&[0, 1, 2, 3, 4, 5]),
                    vec![link(0, 1), link(2, 3)],
                    20
                ),
                ( // M(A-B-A)CDE
                    occupation(&[0, 1, 0, 2, 3, 4]),
                    vec![link(0, 1), link(1, 2)],
                    9
                ),
                ( // M(A-B-C)_2
                    occupation(&[0, 1, 2, 0, 1, 2]),
                    vec![link(0, 1), link(1, 2), link(3, 4), link(4, 5)],
                    11
                ),
                ( // M(A-B-B-A)CD
                    occupation(&[0, 1, 1, 0, 2, 3]),
                    vec![link(0, 1), link(1, 2), link(2, 3)],
                    7
                ),
                ( // M(A-B-C-B-A)D
                    occupation(&[0, 1, 2, 1, 0, 3]),
                    vec![link(0, 1), link(1, 2), link(2, 3), link(3, 4)],
                    7
                )
            ]),
        ];

        for (name, shape_tests) in tests {
            let shape = shape_from_name(name);
            for (occupation, links, expected_count) in shape_tests {
                let base = Coordination::new_unsorted(occupation.clone(), links.clone());
                let distinct = base.rotationally_distinct_permutations(shape);

                // Remove distincts with trans-aligned links for the sake of counting
                let without_trans_links: Vec<_> = distinct.iter()
                    .filter(|(coordination, _)| {
                        coordination.links.iter()
                            .all(|link| link_angle(link, shape) < 0.95 * std::f64::consts::PI)
                    })
                    .collect();

                let count = without_trans_links.len();
                if count != expected_count {
                    println!("{} with occupation {:?} and links {:?} found {} distinct, not {} as expected", name, occupation.sigma, links, count, expected_count);
                    for (coordination, _) in without_trans_links {
                        println!("- {}", coordination);
                    }
                    panic!();
                }
            }
        }
    }
}
