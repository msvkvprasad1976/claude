#!/usr/bin/env bash
# Full protocol. Each run logs to its own folder. Aggregator builds the tables.
set -e
SEEDS="0 1 2 3 4"
EP=200

echo "### ModelNet40 (PointNet++)"
for s in $SEEDS; do
  python train_modelnet40.py --weighting cotangent     --baseline ddg     --epochs $EP --seed $s
  python train_modelnet40.py --weighting combinatorial --baseline ddg     --epochs $EP --seed $s
  python train_modelnet40.py --baseline l2      --epochs $EP --seed $s
  python train_modelnet40.py --baseline dropout --epochs $EP --seed $s
  python train_modelnet40.py --baseline none    --epochs $EP --seed $s
done

echo "### QM9 (GIN)"
for s in $SEEDS; do
  python train_qm9.py --baseline ddg  --epochs $EP --seed $s
  python train_qm9.py --baseline l2   --epochs $EP --seed $s
  python train_qm9.py --baseline none --epochs $EP --seed $s
done

echo "### Aggregate"
python aggregate_results.py
echo "DONE. See ./outputs/tables/summary.md"
