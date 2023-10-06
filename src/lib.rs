#![warn(missing_docs)]

//! General Molecule-modeling with full stereochemistry
//!
//! This crate aims to provide a Molecule class that is suitable for both organic and inorganic
//! molecular modeling incorporating stereochemistry in a general fashion for any coordination
//! polyhedron.

#[macro_use]
extern crate lazy_static;

/// Struct for representing order permutations and helper functions
pub mod permutation;

/// Representation of a coordination polyhedron
pub mod shapes;

/// Helper methods for working with quaternion fits
pub mod quaternions;

/// Strong types for working with disjoint index spaces
pub mod strong;

/// Various lower-level geometric helper functions, e.g. signed angles and plane fits
pub mod geometry;

/// Distance Geometry module, for generating new 3D embeddings
pub mod dg;

/// Moments and axes of rotational inertia
pub mod inertia;

/// Stereopermutation
pub mod stereo;

/// Molecule representation
pub mod molecule;
