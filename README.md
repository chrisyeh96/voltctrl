# Online learning for robust voltage control under uncertain grid topology

[Christopher Yeh](https://chrisyeh96.github.io/), [Jing Yu](https://scholar.google.com/citations?user=akiDVE8AAAAJ&hl=en), [Yuanyuan Shi](https://yyshi.eng.ucsd.edu/), [Adam Wierman](https://adamwierman.com/)
<br>**California Institute of Technology** and **UC San Diego**

This repo contains code for the following two papers:

**Robust online voltage control with an unknown grid topology**
<br>C. Yeh, J. Yu, Y. Shi, and A. Wierman
<br>ACM e-Energy 2022, **Best paper award finalist**
<br>[**Paper**](https://dl.acm.org/doi/10.1145/3538637.3538853) |
[**Video**](https://youtu.be/iDhDfDrXqoA)

**Online learning for robust voltage control under uncertain grid topology**
<br>C. Yeh, J. Yu, Y. Shi, and A. Wierman
<br>Under submission
<br>[**Preprint**](https://arxiv.org/abs/2306.16674)


## Getting started

The code and instructions in this repo were tested on an Amazon AWS EC2 `m5ad.8xlarge` (32 CPU cores, 128 GiB RAM) instance running Ubuntu 22.04 LTS.

### Install packages

1. Install [miniconda3](https://docs.conda.io/en/latest/miniconda.html). Once miniconda3 is installed, we recommended that you use the `libmamba` solver for faster conda dependency resolution:
    ```bash
    conda config --set solver libmamba
    ```
2. Install the `voltctrl` conda environment:
    ```bash
    conda env update -f env.yml --prune
    ```
3. Request a Mosek license ([link](https://www.mosek.com/products/academic-licenses/)). Upon receiving the license file (`mosek.lic`) in an email, create a folder `~/mosek` and copy the license file into that folder.

### Running voltage control experiments

The main two scripts are [run.py](run.py), which simulates bus voltages under approximate linearized distribution grid dynamics, and [run_nonlinear.py](run_nonlinear.py), which simulatves bus voltages with a nonlinear balanced AC single-phase model.

## Data Files (in `/data`)

The original data files were provided by the authors of the following paper:
> Guannan Qu and Na Li. 2020. Optimal Distributed Feedback Voltage Control under Limited Reactive Power. _IEEE Transactions on Power Systems_ 35, 1 (Jan. 2020), 315–331. https://doi.org/10.1109/TPWRS.2019.2931685

The original data files ("orig_data.zip") are attached to the [releases](https://github.com/chrisyeh96/voltctrl/releases/tag/v1.0). These original data files have been processed into the following files, which are the main files relevant for our experiments. See the [inspect_matlab_data.ipynb](notebooks/inspect_data.ipynb) notebook for details.

**[PV.mat](data/PV.mat)**
- contains single key `'actual_PV_profile'`
- float64 array, shape [14421]
- min: 0.0, max: ~13.4
- units: MW
- description: solar generation, measured every 6 seconds

**[aggr_p.mat](data/aggr_p.mat)**
- contains single key `'p'`
- float64 array, shape [14421]
- min: ~2.4, max: ~7.1
- units: MW
- description: active power load, measured every 6 seconds for 24h

**[aggr_q.mat](data/aggr_q.mat)**
- contains single key `'q'`
- float64 array, shape [14421]
- min: ~1.1, max: ~3.1
- units: MVar
- description: reactive power load, measured every 6 seconds for 24h

**[pq_fluc.mat](data/pq_fluc.mat)**
- contains single key `'pq_fluc'`
- float64 array, shape [55, 2, 14421]
- units: MW for p, MVar for q
- for p, min: ~-0.9, max: ~3.7
- for q, min: ~-0.5, max: ~0.0
- description: active and reactive power injection at 55 buses, measured every 6 seconds for 24h
  - first column is p, second column is q
  - + means generation, - means load
  - p is net active power injection (solar generation - load)

**[SCE_56bus.mat](data/SCE_56bus.mat)**
- contains single key `'case_mpc'`
- description: a "MATPOWER" file
- `mat['case_mpc'][0,0]` has 4 "keys"
    - 'version': shape [1], type uint8
    - 'baseMVA': shape [1, 1], type uint8, reference voltage at root bus
    - 'bus': shape [56, 13], type float64
    - 'branch': shape [55, 13], type float64
    - 'gen': shape [1, 21], type int16

**[nonlinear_voltage_baseline.npy](data/nonlinear_voltage_baseline.npy)**
- float64 array, shape [14421, 56]
- description: balanced AC nonlinear simulation voltages, generated by [nonlinear_no_control.py](nonlinear_no_control.py)
- each column is the voltage of a bus, with column 0 being bus 0 (the substation)
- units: p.u. voltage (multiply by 12 to get kV)

See the attachments in [releases](https://github.com/chrisyeh96/voltctrl/releases/) for Python `.pkl` files containing the results of running the various algorithms. These Pickle files are read by the various Jupyter notebooks in the [notebooks](notebooks/) folder for plotting and analysis.


## Citation

Please cite our papers as follows, or use the BibTeX entries below.

> C. Yeh, J. Yu, Y. Shi, and A. Wierman, "Robust online voltage control with an unknown grid topology," in _Proceedings of the Thirteenth ACM International Conference on Future Energy Systems (e-Energy '22)_, Association for Computing Machinery, Jun. 2022, pp. 240–250, ISBN: 9781450393973. DOI: 10.1145/3538637.3538853. [Online]. Available: [https://dl.acm.org/doi/10.1145/3538637.3538853](https://dl.acm.org/doi/10.1145/3538637.3538853).
>
> C. Yeh, J. Yu, Y. Shi, A. Wierman, "Online learning for robust voltage control under uncertain grid topology," Jun. 2023. DOI: 10.48550/arXiv.2306.16674. [Online]. Available: [https://arxiv.org/abs/2306.16674](https://arxiv.org/abs/2306.16674).

```tex
@inproceedings{
    yeh2022robust,
    author = {Christopher Yeh and Jing Yu and Yuanyuan Shi and Adam Wierman},
    booktitle = {{Proceedings of the Thirteenth ACM International Conference on Future Energy Systems (e-Energy '22)}},
    doi = {10.1145/3538637.3538853},
    isbn = {9781450393973},
    month = {6},
    pages = {240-250},
    publisher = {Association for Computing Machinery},
    title = {Robust online voltage control with an unknown grid topology},
    url = {https://dl.acm.org/doi/10.1145/3538637.3538853},
    year = {2022}
}

@article{yeh2023online,
    author={Yeh, Christopher and Christianson, Nicolas and Low, Steven and Wierman, Adam and Yue, Yisong},
    month={6},
    title={{Online learning for robust voltage control under uncertain grid topology}},
    url={https://arxiv.org/abs/2306.16674},
    doi={10.48550/arXiv.2306.16674},
    year={2023}
}
```
