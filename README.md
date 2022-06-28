[**Paper**](https://dl.acm.org/doi/10.1145/3538637.3538853) |
[**Video**](https://youtu.be/iDhDfDrXqoA)

# Robust Online Voltage Control with an Unknown Grid Topology

[Christopher Yeh](https://chrisyeh96.github.io/), [Jing Yu](https://scholar.google.com/citations?user=akiDVE8AAAAJ&hl=en), [Yuanyuan Shi](https://yyshi.eng.ucsd.edu/), [Adam Wierman](https://adamwierman.com/)

**California Institute of Technology and UC San Diego**


## Data Files (in `/data`)

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
- description: active power injection, measured every 6 seconds for 24h

**aggr_q.mat**
- contains single key `'q'`
- float64 array, shape [14421]
- min: ~1.1, max: ~3.1
- units: MVar
- description: reactive power injection, measured every 6 seconds for 24h

**pq_fluc.mat**
- contains single key `'pq_fluc'`
- float64 array, shape [55, 2, 14421]
- units: MW for p, MVar for q
- for p, min: ~-0.9, max: ~3.7
- for q, min: ~-0.5, max: ~0.0
- description: p,q variation (sum of solar + load) at 55 buses, measured every 6 seconds for 24h
  - both p and q are power injection (+ means generation, - means load)

**SCE_56bus.mat**
- contains single key `'case_mpc'`
- description: a "MATPOWER" file. See the convert2matpower.m file for description.
- mat['case_mpc'][0,0] has 4 "keys"
    - 'version': shape [1], type uint8
    - 'baseMVA': shape [1, 1], type uint8, reference voltage at root bus
    - 'bus': shape [56, 13], type float64
    - 'branch': shape [55, 13], type float64
    - 'gen': shape [1, 21], type int16

See the attachment attached to [releases](https://github.com/chrisyeh96/voltctrl/releases) for Python `.pkl` files containing the results of running the various algorithms. These Pickle files are read by the various Jupyter notebooks for plotting and analysis.


## Citation

Please cite this article as follows, or use the BibTeX entry below.

> C. Yeh, J. Yu, Y. Shi, and A. Wierman, "Robust online voltage control with an unknown grid topology," in _e-Energy '22: Proceedings of the Thirteenth ACM International Conference on Future Energy Systems_, Association for Computing Machinery, Jun. 2022, pp. 240–250, ISBN: 9781450393973. DOI: 10.1145/3538637.3538853. [Online]. Available: [https://dl.acm.org/doi/10.1145/3538637.3538853](https://dl.acm.org/doi/10.1145/3538637.3538853).

```tex
@inproceedings{
    yeh2022robust,
    author = {Christopher Yeh and Jing Yu and Yuanyuan Shi and Adam Wierman},
    booktitle = {{e-Energy 22': Proceedings of the Thirteenth ACM International Conference on Future Energy Systems}},
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
