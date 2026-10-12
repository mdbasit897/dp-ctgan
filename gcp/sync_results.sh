#!/usr/bin/env bash
# Copy results to a PRIVATE Cloud Storage bucket (run on the VM). Create the bucket once:
#   gcloud storage buckets create gs://YOUR_BUCKET --location=us-central1 --uniform-bucket-level-access
# MIMIC-III-derived files (synthetic_*.csv.gz under results/mimic3) fall under the PhysioNet DUA:
# keep the bucket private, or add --exclude to skip them.
set -euo pipefail
BUCKET="${BUCKET:-gs://YOUR_BUCKET}"
cd "$(dirname "$0")/.."
gcloud storage rsync --recursive results "$BUCKET/results"
gcloud storage rsync --recursive logs "$BUCKET/logs"
echo "Synced to $BUCKET. Download to your laptop with: gcloud storage rsync --recursive $BUCKET/results ./results"
