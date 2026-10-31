# ERP decoding 笔记


## Step 1：下载数据
- 脚本：step1_download.py（hu-neuro-pipeline 0.9.0，pipeline.datasets.get_erpcore）
- ERP CORE N170（人脸知觉任务），OSF 项目 pfde9，只下 sub-001
- 存放：data/step1_erpcore_N170/erpcore/N170/（BIDS 结构）
- sub-001：33 通道，1024 Hz，约 683 s，640 个事件标记

## Step 2：预处理（眼电）
- 脚本：step2_preprocess.py
- HEOG_left、HEOG_right、VEOG_lower 标成 EOG 类型（.set 读进来全是 EEG）
- 双极眼电：HEOG = HEOG_left − HEOG_right，VEOG = VEOG_lower − FP2（原通道保留，FP2 仍是 EEG）
- 输出：data/step2_eog-raw.fif（35 通道：30 EEG + 5 EOG，640 个标记）
