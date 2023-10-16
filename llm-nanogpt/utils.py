import sys
from packaging import version

import torch

def is_pytorch2():
  return version.parse(torch.__version__) >= version.parse("2.0.0")

def is_support_compile():
  return is_pytorch2() and sys.version_info < (3, 11)



