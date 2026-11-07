# Third-party notices

This project uses the following open-source components, unmodified, under their own licences.

| Component | Use | Licence |
|---|---|---|
| [gsplat](https://github.com/nerfstudio-project/gsplat) | Gaussian rasterisation and densification on GPU | Apache-2.0 |
| [COLMAP](https://colmap.github.io/) / [pycolmap](https://github.com/colmap/pycolmap) | Structure-from-Motion | BSD-3-Clause |
| [OpenCV](https://opencv.org/) | Image processing, ORB, ArUco, YuNet runtime | Apache-2.0 |
| [YuNet face detector](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | Face anonymisation (model downloaded at runtime) | MIT |
| [PyTorch](https://pytorch.org/) | GPU training | BSD-3-Clause |
| NumPy, SciPy, Matplotlib, PyYAML, plyfile, Typer, Rich | Numerics, plots, config, CLI | BSD / MIT / PSF-compatible |
| [Spark](https://github.com/sparkjsdev/spark) (World Labs) | Gaussian splat rendering in the browser | MIT |
| [three.js](https://threejs.org/) | Browser 3D rendering | MIT |

The original 3D Gaussian Splatting reference implementation (Inria/MPII) is **not** used. Its licence
permits non-commercial research only.

Demo data licences and attribution are listed in `docs/DATA_SOURCES.md`.
