# High level gripes with Molassembler

- For large shapes, all stereopermutations are enumerated, which if asymmetric,
  is expensive in time and space. Avoid by laziness/tricks (via index of
  permutation?) or size-gate
  - Not sure this is avoidable as soon as realizeability comes into play with
    multidentate ligands
- Never fuzzed `Molecule` properly, i.e. generated a set of things you could
  do with it that don't fail and then property tested manipulations of it
- Singletons! No global settings this time
- There are some editing operations that could produce multiple steric
  arrangements, e.g. adding a new ligand to square planar. Previously, an option
  was to receive one of these randomly in single-molecule mutation, but it would
  be cool to have `Mol -> Vec<Mol>` functions instead that do this explicitly
- Ranking! Must be better in terms of
  - Correctness
  - Debuggability (no more `if constexpr` writing files in debug mode, this
    should be a template visitor)
  - Freedom to rank both starting from a vertex, and from an edge


# Ideas

- Parametrizeable shapes (e.g. for symmetric distortions in octahedron or
  bipyramids)
- Shape classification maybe should have included the possibility that a
  coordination does not resemble any of the available shapes, i.e. that a center
  is classed as unstructured
  - Upside might be that this could really help with "distorted" ideal shapes in
    small cycles etc.
  - Downside to this is that spatial modeling probably trips over this hard and
    unstructuredness has a non-local messiness knock-on effect
- Add interface to semiempirics: yaehmop / Sparrow (MNDO/PM6)
  - Use for cheap evaluations of single-center best shape by cutting the
    molecule into a fragment (to a cutoff distance, with possibly more if
    there's a metal)
  - Use for evaluation of rotational minima over builtin
    - Must have very rigid, reproducible steric model of fragment for evaluation
      - Ideally a geometry optimization at every tested angle (with dihedral
        fixing only)
      - Instead of sampling fixed intervals all around the angle, try smarter
        methods to find minima as soon as some reasonably accurate curves are
        available to test against
    - Makes serialization challenging, though!
- Structural interpretation: Try assembling a molecule without going for bond
  orders at all, but only figuring out the best shape that matches a local
  neighborhood and distances
- ML shape predictor (i.e. classify vertex of subgraph into shape)
- Bond stereopermutator could be generalized to allene case (i.e. placed on
  non-directly connected atoms)
- Try arrayfire to get single codebase for CPU/GPU DG

look through the GitLab issues again!


# Dynamic shape list

Think about this again, and hard! Is this really wanted?

## Problems

- [x] Limit information necessary to add shapes to lower barrier to entry
  - [x] Rotations
  - [x] Mirror
  - [x] Tetrahedra
- [ ] Serialization (dynamically added shapes have to be part of the format)
- [ ] Parallelism (is the list of shapes mutable and does it have to be fenced?)
  e.g. in workflows importing serializations, new shapes may be encountered. are
  they supposed to be added to the list of shapes? Is it part of the molecule
  data?

## Plan

- rename `Name` to `StaticName` (or `BuiltIn` or so) with LSP and add

  ```rust
  #[derive(PartialEq, Eq, Clone, Debug, Hash)]
  pub enum Name {
      Static(StaticName),
      Dynamic(String)
  }

  impl std::fmt::Display for Name {
      fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
          match self {
              Name::Static(name) => write!(f, "{}", name),
              Name::Dynamic(name) => write!(f, "{}", name)
          }
      }
  }

  // Some converting helpers for 
  // - `StaticName::Tetrahedron.into()` instead of
  // - `Name::Static(StaticName::Tetrahedron)`
  impl From<StaticName> for Name {
      ...
  }

  impl From<String> for Name {
      ...
  }

  impl From<&'static str> for Name {
      ...
  }

  ```

- Then visit all instances of `StaticName` and fix the complications


# Molecule data type

1. Graph (w/out stereo) x Stereo HashMaps x Regularization
   + Algorithms acting only on Graph can take &Graph
   + Initial type of reading data is Graph, library consumers can stop
     there if they want and avoid all the stereo interpretation
2. Graph (w/ stereo) x Regularization
   + Only a single type for everything to act on and all the relevant
     data is always present
   - Probably (?) less space efficient as there are a bunch of Nones
     on terminal atoms, of which there are usually many in a molecular
     graph
   - To figure out the state of if stereo data has been added, either
     have to add a bool to the data representation or look for a
     non-terminal atom for if data is present


# To do

- Closer refinement errf variation speed and correctness checking (e.g. compare
  `f32` and `f64` variants, too)
- Similarity matching could be generalized to M <= N so that we can find shape
  transitions directly (see molassembler's vertex matching subgraph MR)
- Check shape methods
  - [x] rotation finding
  - [x] superposable vertex sets
  - [ ] canonicalization
    - `canonicalize_coordinates` isn't there yet, need to rethink pivot point
      selection in the chosen plane since random selection works for planes that are
      completely equivalent by rotation around +z, but not many shapes have that
      property - need to find a 'leftmost' property for asymmetric cases
  - [ ] tetrahedra (buggy!)
  - [ ] mirror
- Is `shape::recognition::sample` correct? shouldn't it distort more than just
  scaling vertices, but in random directions?
- Is it okay to unit rescale vectors just so that `scaling::minimize` has a
  simple domain to minimize over?
- Have another close look at the arbitrary failure thresholds in the similarity
  matching tricks binary. Maybe it's better not just to take the smallest one
  you found but look at the percentage of failures at each value of the
  quaternion fit msd and model it as a cumulative distribution function. That
  way you could choose what probability of failure you're willing to accept,
  making for a more meaningful choice of threshold.
- Split into crates: shape, stereo, molecule?
