# 基于 ERP CORE N170 单被试脑电的人脸与汽车知觉时间分辨解码

完整笔记（中文 / English）：[ERP_notes.html](ERP_notes.html)

## 文件

| 步骤 | 文件 |
|---|---|
| Step 1 下载数据 | `step1_download.py` |
| Step 2 预处理 | `step2_preprocess.py`、`step2_check_oz.py`、`step2_check_ica003.py` |
| Step 3 传统 ERP | `step3_erp.py` |
| Step 4 逐时间点解码（方法 1，CPU） | `step4_lib.py`、`step4_run.py`、`step4_decoding.py` |
| Step 5 置换检验（CPU） | `step5_permutation.py` |
| Step 6 GPU 版 SVM，方法 1 的正式结果 | `step6_gpu_lib.py`、`step6_check.py`、`step6_main.py`、`step6_permutation.py` |
| Step 7 原始数据版本 | `step7_raw.py` |
| Step 8 时间窗解码（方法 2） | `step8_window.py` |

- 过程记录：`notes.md`；参考文献：`references.md`；图：`figs/`
- 环境：`requirements.txt`
- 数据：ERP CORE N170（OSF 项目 [pfde9](https://osf.io/pfde9/)），只用了被试 sub-001；数据不在仓库里，运行 `step1_download.py` 下载
