# Shared runtime for V2X-WAM training, evaluation, and data scripts.
# Source this file; do not execute it.
V2X_WAM_ENV="/mnt/nvme1n1/wlz/env/conda_envs/v2x-wam"
V2X_WAM_PYTHON="${V2X_WAM_ENV}/bin/python"
V2X_WAM_PROJECT="/mnt/nvme1n1/LCX2/nyc/FADO-WAM"
V2X_WAM_WORKSPACE="/mnt/nvme1n1/LCX2/nyc/fado-wam_workspace"
V2X_WAM_DATA="/mnt/nvme1n1/LCX2/nyc/data/VLA4CoDrive"
V2X_WAM_SPD="/mnt/nvme1n1/LCX2/nyc/data/V2X-Seq-SPD"

if [[ ! -x "${V2X_WAM_PYTHON}" ]]; then
  echo "v2x-wam python not found: ${V2X_WAM_PYTHON}" >&2
  exit 1
fi

export PATH="${V2X_WAM_ENV}/bin:${PATH}"
export PYTHON="${V2X_WAM_PYTHON}"
