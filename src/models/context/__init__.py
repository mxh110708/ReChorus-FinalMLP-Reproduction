from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

from .FinalMLP_ReImpl import FinalMLPReImplTopK, FinalMLPReImplCTR
__all__ += ['FinalMLPReImplTopK', 'FinalMLPReImplCTR']

# ⭐ 关键：给模块起个没有下划线的别名，配合 main.py 的 eval 规则
from . import FinalMLP_ReImpl as FinalMLPReImpl
__all__ += ['FinalMLPReImpl']

