# Pinned reference material

`upstream.json` records exact repository revisions and SHA256 hashes. `reference_sources.zip` contains only the reviewed source files needed to trace the map, initial seed, transition model, methods, and renderer interface. It is reference material, not automatically imported application code.

- SNO-MDP: https://github.com/akifumi-wachi-4/safe_near_optimal_mdp
- SafeMDP: https://github.com/befelix/SafeMDP
- Copyright/permission notices: `LICENSE.snomdp` and `LICENSE.safemdp`.
- Released map: `../data/gp_safety_gym/released_map.npz`, copied unchanged from the recorded SNO-MDP revision.

The original viewer requires legacy `gym`, `safety_gym`, and `mujoco_py`; those packages are not installed or executed by the headless project. The archived renderer is not a drop-in modern MuJoCo integration. Do not call the browser replay native Engine_GP execution.

Two explicitly reviewed pure helpers are exercised by source-equivalence tests. The tests extract only the named function definitions rather than importing or running upstream drivers.
