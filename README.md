## Data Files (in /data)

**PV.mat**
- contains single key `'actual_PV_profile'`
- float64 array, shape [14421]
- min: 0.0, max: ~13.4
- units: TODO
- TODO: sign
- description: solar generation, measured every 6 seconds

**aggr_p.mat**
- contains single key `'p'`
- float64 array, shape [14421]
- min: ~2.4, max: ~7.1
- units: MW
- description: active power injection, measured every 6 seconds for 24h, TODO

**aggr_q.mat**
- contains single key `'q'`
- float64 array, shape [14421]
- min: ~1.1, max: ~3.1
- units: MVar
- description: reactive power injection, measured every 6 seconds for 24h, TODO

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
- units: TODO
- description: a "MATPOWER" file. See the convert2matpower.m file for description.
- mat['case_mpc'][0,0] has 4 "keys"
    - 'version': shape [1], type uint8
    - 'baseMVA': shape [1, 1], type uint8, reference voltage at root bus
    - 'bus': shape [56, 13], type float64
    - 'branch': shape [55, 13], type float64
    - 'gen': shape [1, 21], type int16
