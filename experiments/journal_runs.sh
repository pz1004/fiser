#!/usr/bin/env bash
# Canonical runs behind every result of the paper; they write results/journal/.
# Each stage stays under 2 h on 24 workers; run them one after another:
#   ./experiments/journal_runs.sh ground | meshes | sat_usnet | sat_small | sat_weather | sat_misc1 | sat_misc2 | sat_new | derived
# The derived stage needs the ground and satellite stages. Runtimes in the logs depend on the machine.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python -u"
R=results/journal
SG=$R/spaceground
mkdir -p $SG
SAT="$PY experiments/run_spaceground.py --constellations 144/12 --step-s 300 --workers 24"
GROUND="$PY experiments/run_ground.py --topology nsfnet cost239 usnet --workers 24"
CORE="B1 B2 B3 B4 B5 H FiSER"
SENS="B1 B2 B3 B5 B7 FiSER"

case "${1:-}" in
  ground)      # single and two-failure protection, memory coherence, sensitivity (about 15 min)
    $GROUND --out $R/ground > $R/ground.log 2>&1
    $GROUND --methods B7 FiSER --out $R/ground_B7 > $R/ground_B7.log 2>&1
    $GROUND --paths 3 --methods $CORE --out $R/ground_alpha2 > $R/ground_alpha2.log 2>&1
    for T in 1 0.1 0.01; do
      $GROUND --methods $CORE --param t_coh_s=$T --out $R/ground_T$T > $R/ground_T$T.log 2>&1
    done
    $GROUND --methods $SENS --param attempts=25 --out $R/ground_sens_M25 > $R/sens_M25.log 2>&1
    $GROUND --methods $SENS --param attempts=100 --out $R/ground_sens_M100 > $R/sens_M100.log 2>&1
    $GROUND --methods $SENS --max-link-km 50 --out $R/ground_sens_scale50 > $R/sens_scale50.log 2>&1
    $GROUND --methods $SENS --max-link-km 150 --out $R/ground_sens_scale150 > $R/sens_scale150.log 2>&1
    $GROUND --methods B1 B2 B3 B4 B5 FiSER --param lam_swap=0.977 --out $R/ground_sens_lam0977 > $R/sens_lam0977.log 2>&1
    $GROUND --methods B1 B2 B3 B4 B5 FiSER --param l_dec_km=2000 --out $R/ground_sens_ldec2000 > $R/sens_ldec2000.log 2>&1
    $GROUND --methods B1 B2 B3 B4 B5 FiSER --param l_dec_km=1e9 --out $R/ground_sens_ldecinf > $R/sens_ldecinf.log 2>&1
    ;;
  meshes)      # random meshes: Table V, B5 study (timings need an otherwise idle machine)
    $PY experiments/run_scaling.py --workers 24 --out $R/scaling > $R/scaling.log 2>&1
    $PY experiments/run_scaling.py --workers 24 --K 5 3 --out $R/scaling_fast > $R/scaling_fast.log 2>&1
    $PY experiments/run_scaling.py --workers 24 --paths 3 --out $R/scaling_alpha2 > $R/scaling_alpha2.log 2>&1
    $PY experiments/run_scaling.py --workers 24 --sizes 200 400 --max-pair-km 200 --out $R/scaling_large > $R/scaling_large.log 2>&1
    $PY experiments/run_b5_study.py --scenario hetero --spreads 0 0.01 0.02 --workers 24 --out $R/b5_study > $R/b5_hetero.log 2>&1
    $PY experiments/run_b5_study.py --scenario mesh --degrees 3.5 3.0 --fth 0.75 0.80 --workers 24 --out $R/b5_study > $R/b5_mesh.log 2>&1
    $PY experiments/run_b5_study.py --scenario alpha2 --workers 24 --out $R/b5_study > $R/b5_alpha2.log 2>&1
    ;;
  sat_usnet)   # about 75 min
    $SAT --topology usnet --ogs 0 2 4 6 --fth 0.80 0.85 0.90 --out $SG/main_usnet > $SG/main_usnet.log 2>&1
    ;;
  sat_small)   # about 25 min
    for t in nsfnet cost239; do
      $SAT --topology $t --ogs 0 2 4 6 --fth 0.80 0.85 0.90 --out $SG/main_$t > $SG/main_$t.log 2>&1
    done
    ;;
  sat_weather) # about 40 min
    for t in nsfnet cost239 usnet; do
      $SAT --topology $t --ogs 6 --fth 0.80 0.85 0.90 --methods FiSER-W --out $SG/weather_$t > $SG/weather_$t.log 2>&1
    done
    $SAT --topology nsfnet --ogs 6 --fth 0.85 --methods H FiSER --out $SG/heuristic_nsfnet > $SG/heuristic_nsfnet.log 2>&1
    ;;
  sat_misc1)   # constellation size and baselines with satellites (NSFNET, four OGS)
    $PY experiments/run_spaceground.py --topology nsfnet --ogs 4 --constellations 72/6 288/18 --fth 0.80 0.85 0.90 \
        --step-s 300 --workers 24 --out $SG/constellation_nsfnet > $SG/constellation_nsfnet.log 2>&1
    $SAT --topology nsfnet --ogs 0 4 --fth 0.80 0.85 0.90 --methods FiSER B1 B2 B5 --out $SG/baselines_nsfnet \
        > $SG/baselines_nsfnet.log 2>&1
    ;;
  sat_misc2)   # hardware sensitivity (NSFNET, four OGS)
    $SAT --topology nsfnet --ogs 0 4 --fth 0.80 0.85 0.90 --sat F_sat=0.87 --out $SG/sens_fsat087 > $SG/sens_fsat087.log 2>&1
    $SAT --topology nsfnet --ogs 0 4 --fth 0.80 0.85 0.90 --sat eta_sys=0.06 --out $SG/sens_etasys006 > $SG/sens_etasys006.log 2>&1
    $SAT --topology nsfnet --ogs 0 4 --fth 0.80 0.85 0.90 --sat attempts=5900 --out $SG/sens_m5900 > $SG/sens_m5900.log 2>&1
    ;;
  sat_new)     # two failures, finite memory coherence, and links lasting at least a minute (NSFNET, six OGS)
    $SAT --topology nsfnet --ogs 0 6 --fth 0.80 0.85 0.90 --paths 3 --methods FiSER B5 --out $SG/alpha2_nsfnet \
        > $SG/alpha2_nsfnet.log 2>&1
    $SAT --topology nsfnet --ogs 0 6 --fth 0.80 0.85 0.90 --param t_coh_s=0.1 --out $SG/T0.1_nsfnet \
        > $SG/T0.1_nsfnet.log 2>&1
    $SAT --topology nsfnet --ogs 6 --fth 0.80 0.85 0.90 --sat min_lifetime_s=60 --out $SG/lifetime60_nsfnet \
        > $SG/lifetime60_nsfnet.log 2>&1
    ;;
  derived)     # model validation, satellite statistics, link lifetimes, weather outages (about 20 min)
    $PY experiments/validate_model.py --runs $R/ground $R/ground_T1 $R/ground_T0.1 $R/ground_T0.01 --mc-designs 30 \
        > $R/validate_model.txt
    $PY experiments/satellite_stats.py > $SG/stats.txt
    for t in nsfnet cost239 usnet; do
      $PY experiments/link_lifetime.py --topology $t > $SG/link_lifetime_$t.txt
    done
    $PY experiments/analyze_weather_outages.py --root $SG
    ;;
  *)
    echo "usage: $0 ground|meshes|sat_usnet|sat_small|sat_weather|sat_misc1|sat_misc2|sat_new|derived" >&2
    exit 2
    ;;
esac
echo "stage ${1} done"
