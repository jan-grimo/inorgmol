# (in)orgmol

## Overview

This is an experimental partial reimplementation of
[Molassembler](https://github.com/qcscine/molassembler) with multiple aims:

- Try out additional features missing from original
  - GPU-offload parts of conformer generation's distance geometry refinement
  - Parameterizeable shapes (e.g. symmetric distortions in octahedron or
    bipyramids)
- Blank slate offers easy path to trying out fixes to high-level gripes
  - Fixed shape list could possibly be dynamic
  - Avoid enumerating all possible atom-centered stereo arrangements when
    interpreting 3D positions
  - Bond-centric stereo arrangements could be found with semiempirics instead of
    hamfisted attempts to find general solutions to arbitrary combinations of
    two shapes
  - Underlying graph is not stable (i.e. vertex removal invalidates vertex
    descriptors), leading to much index management and complex manipulation
    functions
- Personal enrichment: Enjoy ergonomics of Rust and its ecosystem of crates


## Plan

This is for fun. Parts of this might be split off and published as crates at
some point.
