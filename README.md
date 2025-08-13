# Jorjin Human Pose Reconstruction

This project, developed by Ching-Jui Wang in collaboration with Jorjin Ltd., aims to detect human motion and reconstruct it within Virtual Reality environments. It addresses the needs of online meetings and education by providing accurate human pose estimation and reconstruction.

The output of this project includes **SMPL-X parameters** and mesh visualizations.

---

## Overview

This project is an extension of the [**simple_romp**](https://github.com/Arthur151/ROMP/tree/master/simple_romp) implementation of **ROMP** (Regression and Ordinal depth Maps for 3D Human Mesh Recovery).

Enhancements include:

- Hand pose detection (addressing the limitation of simple_romp which does not detect hand poses)
- Support for processing video inputs (with or without hand detection)
- Live demo functionality for real-time applications

---

## Features

- **3D Human Mesh Recovery** from monocular images and videos  
- **Hand Pose Detection** integrated into the pipeline  
- **Video Processing** capabilities for batch or real-time input  
- **Live Demo** for quick testing and visualization  
- Outputs include **SMPL-X parameters**, **mesh visualizations**, and **hand detection overlays**

---

## Installation

### Clone the Repository

To get started, clone this repository to your local machine:

```bash
git clone https://github.com/RayWang1226/Jorjin-Human-Pose-Reconstruction.git
```

### Environment Setup

Please follow the environment setup instructions provided in the [**simple_romp** repository](https://github.com/Arthur151/ROMP/tree/master/simple_romp) before running this project.

---

## Usage

### Hand Pose Detection

To detect hand poses from a video input, run:

```bash
python3 handpose_detect.py --video_path <path_to_video> --output_dir <path_to_results>
```

The processed video with hand pose detection will be saved in the specified output directory.

[Watch Demo Video With Hands Detection](demo/output_pose_overlay.mp4)

---

### Processing Video Inputs

- **With Hand Detection:**

```bash
python3 process_video_with_hands.py --video_path <path_to_video> --output_dir <path_to_results>
```

Outputs include SMPL-X mesh visualization, hand detection overlays, and SMPL-X parameters.

[Watch Demo Video With SMPL-X Mesh Reconstruction](demo/output_rendered.mp4)

- **Without Hand Detection:**

```bash
python3 process_video.py --video_path <path_to_video> --output_dir <path_to_results>
```

Outputs include SMPL-X mesh visualization and SMPL-X parameters.

---

### Live Demo

- **Automatic Hand Detection:**

```bash
python3 live_demo.py
```

- **Manual Hand Pose Shifting:**

```bash
python3 live_demo_manual.py
```

For both live demos, you can modify the URL within the script to use any video stream source you prefer.

---

## Citation & Acknowledgements

If you use this project in your research, please cite the original ROMP paper:

```bibtex
@inproceedings{sun2021romp,
  title={ROMP: Monocular, One-stage, Regression of Multiple 3D People},
  author={Sun, Yu and Bao, Qian and Liu, Wu and Fu, Yili and Black, Michael J and Mei, Tao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2021}
}
```