"""Step 1: download ERP CORE N170 (face perception task), participant sub-001.

Data come from OSF project pfde9 via hu-neuro-pipeline.
"""
from pathlib import Path

from pipeline.datasets import get_erpcore
from pipeline.datasets.erpcore import OSF_IDS

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / 'data' / 'step1_erpcore_N170'

assert OSF_IDS['N170'] == 'pfde9', OSF_IDS['N170']

paths = get_erpcore('N170', participants=['sub-001'], path=OUT_DIR)

for kind, files in paths.items():
    for f in files:
        print(f'{kind}: {Path(f).relative_to(ROOT)}')
