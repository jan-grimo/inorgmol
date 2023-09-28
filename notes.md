# High level gripes with Molassembler

- For large shapes, all stereopermutations are enumerated, which if asymmetric,
  is expensive in time and space. Avoid by laziness/tricks (via index of
  permutation?) or size-gate
  - Not sure this is avoidable as soon as realizeability comes into play with
    multidentate ligands
- Never fuzzed `Molecule` properly, i.e. generated a set of things you could
  do with it that don't fail and then property tested manipulations of it


# Improvement ideas
- Parametrizeable shapes (e.g. for symmetric distortions in octahedron or
  bipyramids)

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


# To do

- Generalize refinement over floating-point types
  - Missing: `refine<F: Float>`: Maybe needs a PR with argmin or some
    trait specializations
- Documentation (deny undocumented and work through all)
- Check shape methods
  - [x] rotation finding
  - [x] superposable vertex sets
  - [ ] canonicalization
  - [ ] tetrahedra (buggy!)
  - [ ] mirror
- Is `shape::recognition::sample` correct? shouldn't it distort more than just
  scaling vertices, but in random directions?
- Is it okay to unit rescale vectors just so that `scaling::minimize` has a
  simple domain to minimize over?
- Have another close look at the failure thresholds in the similarity matching
  tricks binary. Maybe it's better not just to take the smallest one you found
  but look at the percentage of failures at each value of the quaternion fit
  msd and model it as a cumulative distribution function. That way you could
  choose what probability of failure you're willing to accept.
- `canonicalize_coordinates` isn't there yet, need to rethink pivot point
  selection in the chosen plane since random selection works for planes that are
  completely equivalent by rotation around +z, but not many shapes have that
  property - need to find a 'leftmost' property for asymmetric cases
- Split into three crates
  - Most is shape-related, so rename to e.g. `chem-shape`
  - Then `stereo` or `chem-stereo`
  - Then something molecule related (dg should move here)


# Cargo flamegraph

```
2019  echo -1 | sudo tee /proc/sys/kernel/perf_event_paranoid
2020  CARGO_PROFILE_RELEASE_DEBUG=true cargo flamegraph --unit-test molassembler -- gpu_gradient
```
