# BChS opinion dynamics on modular networks

Monte Carlo (MC) simulations, mean-field ordinary differential equation (ODE) simulations, and an interactive dashboard accompanying:

**Biswas-Chatterjee-Sen (BChS) kinetic exchange opinion model on modular networks**  
Hrishidev Unni, Soumyajyoti Biswas, and Anirban Chakraborti.

**Paper DOI:** `[INSERT PAPER DOI HERE]`

## Files

| File | Purpose |
|---|---|
| `MC_Multigroup_PhaseSpace.m` | Multigroup MC phase-space scan, related to Figure 2. |
| `MC_TwoGroup_PhaseSpace.m` | Two-group MC scans of the bipartite and consensus sectors for Figure 3. |
| `ODE_PhaseSpaceScan_2Group.m` | Two-group ODE phase-space scan for Figure 4. |
| `ODE_Trajectory_2Group.m` | Two-group ODE trajectories for Figure 5. |
| `ODE_PhaseSpaceScan_MultiGroup.m` | Multigroup ODE phase-space scan for Figure 6. |
| `ODE_Trajectory_MultiGroup.m` | Multigroup ODE trajectories for Figure 7. |
| `MC_FiniteSizeScaling.m` | MC simulations underlying Figure 8, varying the number of groups at fixed group size. Saves moments of the community spread needed for susceptibility analysis. |
| `MC_Equilibration_Trajectories.m` | MC time trajectories for checking equilibration. |

## Requirements

- A recent MATLAB release supporting local functions in scripts, `exportgraphics`, and `clim`.
- Parallel Computing Toolbox for the parallel MC scans and multigroup ODE phase-space scan.
- Statistics and Machine Learning Toolbox for `binornd` in the two-group MC and finite-size scripts.
- For the optional dashboard: Python 3.10 or newer, Streamlit, NumPy, and Matplotlib.

## Running the MATLAB simulations

Set the repository folder as MATLAB's current folder. Run a script by its filename without the `.m` extension, for example:

```matlab
ODE_PhaseSpaceScan_2Group
ODE_Trajectory_MultiGroup
MC_FiniteSizeScaling
```

Choose the two-group MC sector using:

```matlab
MC_TwoGroup_PhaseSpace('bipartite')
MC_TwoGroup_PhaseSpace('consensus')
% MC_TwoGroup_PhaseSpace('both') runs both sectors sequentially.
```

Simulation parameters are defined near the top of each file. Full MC scans can take substantial time. The multigroup MC driver starts a six-worker pool and should be launched with no existing parallel pool. Other parallel scripts use their configured worker counts.

MC scans automatically resume available checkpoints. Use a fresh working folder when changing parameters to avoid resuming an incompatible run.

Simulation data and checkpoints are generally saved under `ModelRuns/`. The multigroup MC script saves phase-map images under `Figures/`. The equilibration script saves its `.mat` file and three PNGs in the current folder.

## Citation

If you use this code, cite the accompanying paper using the DOI above.
