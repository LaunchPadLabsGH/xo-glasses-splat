# Data sources and attribution

Raw data is downloaded by `scripts/fetch_demo_data.sh` into `data/raw/` and is never committed. The
published web assets in `docs/assets/` are derived from these sources: faces blurred, reduced to
sparse points and splats.

| Scene | Source | Licence | Attribution | Why it is used |
|---|---|---|---|---|
| `tum-desk` | TUM RGB-D benchmark, `freiburg1_desk` | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers. *A Benchmark for the Evaluation of RGB-D SLAM Systems*, IROS 2012. https://cvg.cit.tum.de/data/datasets/rgbd-dataset | Handheld camera with motion-capture ground truth: real accuracy numbers |
| `dc-glass` | Wikimedia Commons, *Wintry DC through Google Glass* | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | brownpau, https://commons.wikimedia.org/wiki/File:Wintry_DC_through_Google_Glass.webm | Real smart-glasses footage of an urban environment |
| `london-walk` | Wikimedia Commons, *Canada Water and Rafter Walk in Southwark, London Docklands* | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Acabashi, https://commons.wikimedia.org/wiki/File:Canada_Water_and_Rafter_Walk_in_Southwark,_London_Docklands.webm | UK urban walk in 4K |

Derived assets of the `london-walk` scene are shared under CC BY-SA 4.0, as its licence requires. That
licence does not extend to this repository's code.

## Sources considered and not used

| Source | Reason |
|---|---|
| Tanks and Temples | Licence grant limited to non-commercial research |
| Project Aria datasets, Ego4D | Non-commercial licences |
| Mip-NeRF 360, nerfstudio sample data | Licence terms unclear |
