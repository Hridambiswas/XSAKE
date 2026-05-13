from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import yaml


@dataclass
class HADSConfig:
    min_sparsity: float = 0.10
    max_sparsity: float = 0.90
    calibration_steps: int = 10
    block_size: int = 128
    window_size: int = 2
    entropy_temperature: float = 1.0


@dataclass
class KernelConfig:
    block_size_q: int = 128
    block_size_kv: int = 128
    sparsity_type: Literal["dense", "local", "random", "bigbird", "hads"] = "hads"
    use_pallas: bool = True
    hads: HADSConfig = field(default_factory=HADSConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "KernelConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        hads = HADSConfig(**d.pop("hads", {}))
        return cls(**d, hads=hads)


@dataclass
class ModelConfig:
    vocab_size: int = 50_257
    seq_len: int = 1_024
    n_layers: int = 12
    n_heads: int = 12
    d_model: int = 768
    d_ff: int = 3_072
    dropout: float = 0.0
    bias: bool = False
    dtype: str = "bfloat16"
    kernel: KernelConfig = field(default_factory=KernelConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            d = yaml.safe_load(f)
        kernel_d = d.pop("kernel", {})
        hads_d = kernel_d.pop("hads", {})
        kernel = KernelConfig(**kernel_d, hads=HADSConfig(**hads_d))
        return cls(**d, kernel=kernel)

    @property
    def d_head(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class TrainingConfig:
    batch_size: int = 32
    seq_len: int = 1_024
    max_steps: int = 10_000
    warmup_steps: int = 500
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    precision: Literal["float32", "bfloat16"] = "bfloat16"
    gradient_checkpointing: bool = True
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 1_000
    checkpoint_dir: str = "checkpoints"

    @classmethod
    def from_yaml(cls, path: str) -> "TrainingConfig":
        with open(path) as f:
            return cls(**yaml.safe_load(f))


@dataclass
class MeshConfig:
    n_devices: int = 1
    data_axis: str = "data"
    model_axis: str = "model"
    data_parallel: int = 1
    model_parallel: int = 1

    @classmethod
    def from_yaml(cls, path: str) -> "MeshConfig":
        with open(path) as f:
            return cls(**yaml.safe_load(f))
