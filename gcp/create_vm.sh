#!/usr/bin/env bash
# Create a GPU VM for the experiments. Edit the variables, then:  bash gcp/create_vm.sh
# Needs: gcloud CLI logged in (gcloud auth login), billing enabled, and GPU quota in REGION
# (IAM & Admin -> Quotas: "GPUs (all regions)" >= 1 and "NVIDIA L4 GPUs" >= 1 in the region).
set -euo pipefail

PROJECT="${PROJECT:-your-gcp-project-id}"
ZONE="${ZONE:-us-central1-a}"            # any zone that offers L4 GPUs: gcloud compute accelerator-types list --filter="name=nvidia-l4"
VM="${VM:-dpctgan-gpu}"
MACHINE="${MACHINE:-g2-standard-16}"     # 16 vCPU, 64 GB RAM, 1x NVIDIA L4 (24 GB). RF/kNN evaluation is CPU-bound.
DISK_GB="${DISK_GB:-200}"
SPOT="${SPOT:-false}"                    # true = ~60-70% cheaper, but the VM can be preempted (runs are resumable)

EXTRA=()
if [[ "$SPOT" == "true" ]]; then
  EXTRA+=(--provisioning-model=SPOT --instance-termination-action=STOP)
fi

gcloud compute instances create "$VM" \
  --project="$PROJECT" --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --maintenance-policy=TERMINATE \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size="${DISK_GB}GB" --boot-disk-type=pd-balanced \
  --scopes=cloud-platform \
  "${EXTRA[@]}"

echo
echo "VM created. Next:"
echo "  gcloud compute ssh $VM --zone $ZONE --project $PROJECT"
echo "  then on the VM:  bash setup_vm.sh   (copy it with: gcloud compute scp gcp/setup_vm.sh $VM:~ --zone $ZONE)"
