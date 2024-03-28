"""Creates list of resulting pkl files used in IEEE TSG paper.

Usage:
    python generate_pkl_list.py
    zip -9 linear_out.zip -@ < pkl_list_linear.txt
    zip -9 nonlinear_out.zip -@ < pkl_list_nonlinear.txt

The first command
    python generate_pkl_list.py
run this script, which creates two text files, pkl_list_linear.txt and
pkl_list_nonlinear.txt, containing all of the paths to .pkl experiment result
files from the IEEE TSG paper.

The second and third commands
    zip -9 linear_out.zip -@ < pkl_list_linear.txt
    zip -9 nonlinear_out.zip -@ < pkl_list_nonlinear.txt
creates corresponding zip files of all the .pkl files listed in pkl_list_linear.txt
and pkl_list_nonlinear.txt. The -9 option uses the maximum compression rate.
"""

from glob import glob


# ========== linear simulations ==========

pkl_paths = [
    'out/CBCconst_20230809_234150.pkl',  # fixed X̂, fixed etahat
    'out/CBCconst_δ20_η10_20230810_011115.pkl',  # fixed X̂, learned etahat
]

linear_pkl_globs = [
    # known eta
    'out/CBCproj_noise1.0_perm_norm1.0_seed{seed}_2*.pkl',
    'out/CBCproj_noise1.0_perm_norm1.0_seed{seed}_knowntopo14_2*.pkl',
    'out/CBCproj_noise1.0_perm_norm1.0_seed{seed}_knownlines14_2*.pkl',

    # default δ=20
    'out/CBCproj_δ20_η10_noise1.0_perm_norm1.0_seed{seed}_2*.pkl',
    'out/CBCproj_δ20_η10_noise1.0_perm_norm1.0_seed{seed}_knowntopo14_2*.pkl',
    'out/CBCproj_δ20_η10_noise1.0_perm_norm1.0_seed{seed}_knownlines14_2*.pkl',

    # varying δ
    'out/CBCproj_δ1_η10_noise1.0_perm_norm1.0_seed{seed}_knownlines14_2*.pkl',
    'out/CBCproj_δ20_η10_noise1.0_perm_norm1.0_seed{seed}_knownlines14_2*.pkl',
    'out/CBCproj_δ100_η10_noise1.0_perm_norm1.0_seed{seed}_knownlines14_2*.pkl',
    'out/CBCproj_δ500_η10_noise1.0_perm_norm1.0_seed{seed}_knownlines14_2*.pkl',

    # topology change
    'out/CBCproj_δ20_η10_noise1.0_perm_norm1.0_seed{seed}_topochange_2*.pkl',
    'out/CBCproj_δ20_η10_noise1.0_perm_norm1.0_seed{seed}_knowntopo14_topochange_2*.pkl',
    'out/CBCproj_δ20_η10_noise1.0_perm_norm1.0_seed{seed}_knownlines14_topochange_2*.pkl',
]

for search_path in linear_pkl_globs:
    for seed in [8, 9, 10, 11]:
        results = glob(search_path.format(seed=seed))
        num_results = len(results)
        if num_results == 0:
            print('did not find', search_path.format(seed=seed))
        else:
            assert num_results == 1
            pkl_paths.append(results[0])

with open('pkl_list_linear.txt', 'w') as f:
    f.write('\n'.join(pkl_paths))


# ========== nonlinear simulations ==========

outdir = 'out/nonlinear/'

pkl_paths = [
    outdir + 'CBCconst_20230810_130611.pkl',  # fixed X̂, fixed etahat
    outdir + 'CBCconst_δ20_η10_20230810_130842.pkl',  # fixed X̂, learned etahat
]

nonlinear_pkl_globs = [
    # default δ=20
    outdir + 'CBCproj_δ20_η10_noise1.0_perm_norm1.0_seed{seed}_2*.pkl',
    outdir + 'CBCproj_δ20_η10_noise1.0_perm_norm1.0_seed{seed}_knowntopo14_2*.pkl',
    outdir + 'CBCproj_δ20_η10_noise1.0_perm_norm1.0_seed{seed}_knownlines14_2*.pkl',

    # varying δ
    outdir + 'CBCproj_noise1.0_perm_norm1.0_seed{seed}_knownlines14_2*.pkl',
    outdir + 'CBCproj_δ1_η10_noise1.0_perm_norm1.0_seed{seed}_knownlines14_2*.pkl',
    outdir + 'CBCproj_δ20_η10_noise1.0_perm_norm1.0_seed{seed}_knownlines14_2*.pkl',
    outdir + 'CBCproj_δ100_η10_noise1.0_perm_norm1.0_seed{seed}_knownlines14_2*.pkl',
    outdir + 'CBCproj_δ500_η10_noise1.0_perm_norm1.0_seed{seed}_knownlines14_2*.pkl',

    # partial control
    outdir + 'CBCproj_δ20_η10_noise1.0_perm_norm1.0_seed{seed}_partialctrl_2*.pkl',
    outdir + 'CBCproj_δ20_η10_noise1.0_perm_norm1.0_seed{seed}_partialctrl_knownlines14_2*.pkl',
]

for search_path in nonlinear_pkl_globs:
    for seed in [8, 9, 10, 11]:
        results = glob(search_path.format(seed=seed))
        num_results = len(results)
        if num_results == 0:
            print('did not find', search_path.format(seed=seed))
        else:
            assert num_results == 1
            pkl_paths.append(results[0])


with open('pkl_list_nonlinear.txt', 'w') as f:
    f.write('\n'.join(pkl_paths))
