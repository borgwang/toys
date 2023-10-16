from dataclasses import dataclass

from conf.base_conf import BaseConfig

@dataclass
class CommonConfig(BaseConfig):
  base_dir: str = "/data/workspace/tmp"
