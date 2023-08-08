[**Paper**](https://dl.acm.org/doi/10.1145/3538637.3538853) |
[**Video**](https://youtu.be/iDhDfDrXqoA)

# Robust Online Voltage Control with an Unknown Grid Topology

[Christopher Yeh](https://chrisyeh96.github.io/), [Jing Yu](https://scholar.google.com/citations?user=akiDVE8AAAAJ&hl=en), [Yuanyuan Shi](https://yyshi.eng.ucsd.edu/), [Adam Wierman](https://adamwierman.com/)

**California Institute of Technology and UC San Diego**

## Getting started

### Install packages

1. Install [miniconda3](https://docs.conda.io/en/latest/miniconda.html).
2. Install the `voltctrl` conda environment:
    ```bash
    conda env update -f env.yml --prune
    ```
3. Request a Mosek license ([link](https://www.mosek.com/products/academic-licenses/)). Upon receiving the license file (`mosek.lic`) in an email, create a folder `~/mosek` and copy the license file into that folder.

### Running voltage control experiments

TODO

## Data Files (in `/data`)

The original data files were provided by the authors of the following paper:
> Guannan Qu and Na Li. 2020. Optimal Distributed Feedback Voltage Control under Limited Reactive Power. _IEEE Transactions on Power Systems_ 35, 1 (Jan. 2020), 315–331. https://doi.org/10.1109/TPWRS.2019.2931685

The original data files ("orig_data.zip") are attached to the [releases](https://github.com/chrisyeh96/voltctrl/releases/tag/v1.0). These original data files have been processed into the following files, which are the only files relevant for our experiments. See the inspect_matlab_data.ipynb notebook for details.

**PV.mat**
- contains single key `'actual_PV_profile'`
- float64 array, shape [14421]
- min: 0.0, max: ~13.4
- units: MW
- description: solar generation, measured every 6 seconds

**aggr_p.mat**
- contains single key `'p'`
- float64 array, shape [14421]
- min: ~2.4, max: ~7.1
- units: MW
- description: active power load, measured every 6 seconds for 24h

**aggr_q.mat**
- contains single key `'q'`
- float64 array, shape [14421]
- min: ~1.1, max: ~3.1
- units: MVar
- description: reactive power load, measured every 6 seconds for 24h

**pq_fluc.mat**
- contains single key `'pq_fluc'`
- float64 array, shape [55, 2, 14421]
- units: MW for p, MVar for q
- for p, min: ~-0.9, max: ~3.7
- for q, min: ~-0.5, max: ~0.0
- description: active and reactive power injection at 55 buses, measured every 6 seconds for 24h
  - first column is p, second column is q
  - + means generation, - means load
  - p is net active power injection (solar generation - load)

**SCE_56bus.mat**
- contains single key `'case_mpc'`
- description: a "MATPOWER" file
- `mat['case_mpc'][0,0]` has 4 "keys"
    - 'version': shape [1], type uint8
    - 'baseMVA': shape [1, 1], type uint8, reference voltage at root bus
    - 'bus': shape [56, 13], type float64
    - 'branch': shape [55, 13], type float64
    - 'gen': shape [1, 21], type int16

See the attachment in [releases](https://github.com/chrisyeh96/voltctrl/releases/tag/v1.0) for Python `.pkl` files containing the results of running the various algorithms. These Pickle files are read by the various Jupyter notebooks for plotting and analysis.


## Citation

Please cite this paper as follows, or use the BibTeX entry below.

> C. Yeh, J. Yu, Y. Shi, and A. Wierman, "Robust online voltage control with an unknown grid topology," in _Proceedings of the Thirteenth ACM International Conference on Future Energy Systems (e-Energy '22)_, Association for Computing Machinery, Jun. 2022, pp. 240–250, ISBN: 9781450393973. DOI: 10.1145/3538637.3538853. [Online]. Available: [https://dl.acm.org/doi/10.1145/3538637.3538853](https://dl.acm.org/doi/10.1145/3538637.3538853).

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
```
