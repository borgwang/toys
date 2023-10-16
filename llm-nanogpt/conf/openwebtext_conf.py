from dataclasses import dataclass
from conf.common_conf import CommonConfig
from conf.base_conf import BaseConfig


@dataclass
class RunConfig(BaseConfig):
  dataset: str = "openwebtext"
  device: str = "cuda"
  dtype: str = "float16"
  enable_compile: int = 0

  init_from: str = "scratch"
  always_save_checkpoint: bool = True
  log_interval: int = 1

  device_type = "cuda" if "cuda" in device else "cpu"


@dataclass
class TrainingConfig(BaseConfig):
  """owt has ~9B tokens"""
  max_iters:int = 600000
  gradient_accumulation_steps: int = 5 * 8

  # optimizer
  lr: float = 6e-4
  beta1: float = 0.9
  beta2: float = 0.95
  weight_decay: float = 1e-1
  grad_clip: float = 1.0

  # lr schema
  enable_lr_decay: int = 1
  warmup_iters: int = 2000
  lr_decay_iters: int = 600000  # ~= max_iters
  min_lr: float = 6e-5

  # evaluate
  eval_iters: int = 200
  eval_interval: int = 2000


@dataclass
class SampleConfig(BaseConfig):
  start: str = "\n"
  num_samples: int = 10
  max_new_tokens: int = 500
  temperature: float = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
  top_k: int = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
  seed: int = 31


@dataclass
class ModelConfig(BaseConfig):
  context_len: int = 1024
  vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
  n_layer: int = 12
  n_head: int = 12
  n_embed: int = 768
  dropout: float = 0.0
  bias: bool = False


@dataclass
class Config(BaseConfig):
  model: ModelConfig
  training: TrainingConfig
  run: RunConfig
  sample: SampleConfig
  common: CommonConfig

  def setup_finetune(self):
    self.run.always_save_checkpoint: bool = False
    self.run.log_interval = 100

    self.training.lr: float = 3e-5
    self.training.enable_lr_decay: int = 0  # fix lr for finetuning
    self.training.batch_size: int = 2
    self.gradient_accumulation_steps: int = 32
    # 2 micro_batch_size * 32 accumulate_steps * 1024 tokens = 65,536 tokens per iter
    # there are 9B tokens in OWT, 1 epoch is ~150K iters
    self.training.max_iters: int = 200000

    # 8 batches for each evaluate
    self.training.eval_iters: int = self.training.batch_size * self.gradient_accumulation_steps * 8
    self.training.eval_interval: int = 500


cfg = Config(
    model=ModelConfig(),
    training=TrainingConfig(),
    sample=SampleConfig(),
    run=RunConfig(),
    common=CommonConfig()
)
