#!/bin/bash
# Launch XSAKE distributed training on OpenWebText
set -euo pipefail

BACKEND=${BACKEND:-gpu}
STEPS=${STEPS:-10000}
BATCH=${BATCH:-32}
SEQ_LEN=${SEQ_LEN:-1024}
CKPT_DIR=${CKPT_DIR:-checkpoints}

echo "=== XSAKE Training ==="
echo "Backend:   $BACKEND"
echo "Steps:     $STEPS"
echo "Batch:     $BATCH"
echo "Seq len:   $SEQ_LEN"
echo "Checkpoint: $CKPT_DIR"
echo ""

python -c "
from kernels.xla.compiler_flags import setup_xla_flags
setup_xla_flags(backend='$BACKEND')

from training.trainer import train
from model.config import ModelConfig, TrainingConfig, MeshConfig
from data.openwebtext_loader import load_openwebtext
import jax

print(f'Devices: {jax.device_count()} x {jax.devices()[0].device_kind}')

model_cfg    = ModelConfig.from_yaml('configs/training_config.yaml') if False else ModelConfig()
training_cfg = TrainingConfig(
    batch_size=$BATCH, seq_len=$SEQ_LEN,
    max_steps=$STEPS, checkpoint_dir='$CKPT_DIR',
)
mesh_cfg = MeshConfig(n_devices=jax.device_count(),
                      data_parallel=jax.device_count(), model_parallel=1)

data_iter = load_openwebtext(
    split='train', subset_fraction=0.05,
    seq_len=$SEQ_LEN, batch_size=$BATCH,
)

state = train(training_cfg, model_cfg, mesh_cfg, data_iter)
print(f'Training complete. Final step: {int(state.step)}')
"
