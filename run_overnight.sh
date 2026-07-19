#!/bin/bash
# Escalera de convergencia + barrido vs experimento, Frank-Zhang (112).
# Uso:  bash run_overnight.sh
cd "$(dirname "$0")"
set -e
echo "== escalera de convergencia (config ganadora) =="
for G in 28 32 40 48 56; do
    echo "--- grid ${G}^3 ---"
    python3 tools/fit_fz112_bigrid.py $G
done
echo "== barrido de parametros vs experimento a 48^3 =="
python3 tools/scan_fz112_vs_exp.py 48
echo "LISTO. Resultados en outputs/sweep_S/ (P112g*_mu5.npz, scan112_g48.csv)"
