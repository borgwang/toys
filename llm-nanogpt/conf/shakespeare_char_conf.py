from dataclasses import dataclass

from conf.common_conf import CommonConfig
from conf.base_conf import BaseConfig

@dataclass(kw_only=True)
class RunConfig(BaseConfig):
  dataset: str = "shakespeare-char"
  device: str = "cuda"
  dtype: str = "float16"
  enable_compile: int = 0

  init_from: str = "scratch"
  always_save_checkpoint: bool = False
  log_interval: int = 10

  @classmethod
  def __post_init__(cls):
    cls.device_type = "cuda" if "cuda" in cls.device else "cpu"


@dataclass(kw_only=True)
class TrainingConfig(BaseConfig):
  max_iters:int = 5000
  batch_size: int = 64

  gradient_accumulation_steps: int = 1

  # optimizer
  lr: float = 1e-3
  beta1: float = 0.9
  beta2: float = 0.99
  weight_decay: float = 1e-1
  grad_clip: float = 1.0

  # lr schema
  enable_lr_decay: int = 1
  warmup_iters: int = 0
  lr_decay_iters: int = 5000  # ~= max_iters
  min_lr: float = 1e-4

  # evaluate
  eval_iters: int = 200
  eval_interval: int = 250


@dataclass(kw_only=True)
class SampleConfig(BaseConfig):
  start: str = "\n"
  num_samples: int = 10
  max_new_tokens: int = 500
  temperature: float = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
  top_k: int = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
  seed: int = 31


@dataclass(kw_only=True)
class ModelConfig(BaseConfig):
  context_len: int = 256
  vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
  n_layer: int = 6
  n_head: int = 6
  n_embed: int = 384
  dropout: float = 0.2
  bias: bool = False


@dataclass(kw_only=True)
class Config(BaseConfig):
  model: ModelConfig
  training: TrainingConfig
  run: RunConfig
  sample: SampleConfig
  common: CommonConfig


cfg = Config(
    model=ModelConfig(),
    training=TrainingConfig(),
    sample=SampleConfig(),
    run=RunConfig(),
    common=CommonConfig()
)
