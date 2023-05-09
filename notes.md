# High level gripes with Molassembler

- List of recognizable shapes is static
  - Would be cool to further limit the necessary information about shapes to
    lower barrier to entry. Can rotational basis, tetrahedra and mirror be
    inferred from coordinates cheaply?
    - Rotational basis: yes
  - Maybe revisit point group elements and overall symmetries with csm?
- For large shapes, all stereopermutations are enumerated, which if asymmetric,
  is expensive in time and space. Avoid by laziness/tricks (via index of
  permutation?) or size-gate
  - Not sure this is avoidable as soon as realizeability comes into play with
    multidentate ligands

look through the GitLab issues again :D


# TODOs
- Recheck that all uses of new\_random realize that it generates values only in
  [0, 1)