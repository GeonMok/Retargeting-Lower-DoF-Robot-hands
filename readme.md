# Retargeting Lower-DoF Robot Hands with Human Grasping Data

This project aims to generate high-fidelity grasping datasets for lower-DoF robotic hands (specifically focusing on the Allegro Hand) by retargeting human demonstration data from the GRAB dataset.

This repository uses **[AnyTeleop](https://yzqin.github.io/anyteleop/)** and **[dex-retargeting](https://github.com/dexsuite/dex-retargeting)** as its foundational baselines, which are included as submodules.

## Prerequisites & Requirements

To run this project, you will need the following datasets and codebases:

- **Allegro Hand** Kinematics/URDF
- **GRAB Dataset** (Human grasping demonstrations)
- **SMPL-X** (Human body models)
- **dex-retargeting** library

### Environment Setup

This project utilizes two separate Python environments depending on the task:

1. **Retargeting Optimization Environment:**
   For running the kinematic retargeting and trajectory generation processes.

```bash
pip install -r requirements_opt.txt

```

2. **Visualization & Utilities Environment:**
   For rendering visualizations.

```bash
pip install -r requirements.txt

```

## Custom Implementations

While we leverage the `dex-retargeting` library, we have developed several custom modules to improve target generation and optimize the loss function specifically for grasping tasks. All custom implementations are located in the `Codes/` directory.

### Key Custom Files:

- **`custom_allegro_right_position.yml`** A custom configuration file designed to bypass the default heuristic mappings. It enables the pipeline to receive our newly generated continuous targets and perform position-based retargeting tailored for the Allegro right hand.
- **`custom_optimizers.py`** The core of our algorithmic improvements. This file contains custom optimizer classes where we implemented and experimented with new objective functions, including Object-Centric Signed Distance Fields (SDF) and Joint Limit penalties, to ensure physical realism and prevent mesh penetration.
- **`generate_advanced.py`** The main execution script for generating the actual robotic joint trajectories. It utilizes the `custom_optimizers.py` to process the human data and output optimized robotic paths. The generated trajectories can subsequently be visualized and evaluated using `Evaluation/visualize_robot.py`.
