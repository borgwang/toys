from dataclasses import asdict, dataclass

@dataclass
class BaseConfig:
  def update(self, **kwargs):
    for k, v in kwargs.items():
      for sub_cfg_type, sub_cfg_dict in asdict(self).items():
        sub_cfg = getattr(self, sub_cfg_type)
        if k in sub_cfg_dict:
          if sub_cfg_dict[k] != v:
            print(f"overwrite cfg.{sub_cfg_type}.{k} ({sub_cfg_dict[k]} -> {v})")
            setattr(sub_cfg, k, v)
