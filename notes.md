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

### 2.2 平均参考
- set_eeg_reference('average', projection=False)：只用 30 个 EEG 通道求平均，5 个 EOG 通道不参与、也不改变
- 输出：data/step2_avgref-raw.fif
- 验证：每个时间点 30 个电极之和 ≈ 0（最大 0.004 µV，float32 存储误差）；EOG 与 2.1 完全一致
- 60 Hz：参考前功率 3.05 µV²/Hz（相邻频率 2.50），参考后 2.43（相邻 2.41），60 Hz 峰基本消失；原始数据 60 Hz 干扰本来就不大
- EEG 通道平均标准差 338.5 → 166.5 µV（共同的漂移被减掉）

### 2.3.1 高通 0.1 Hz
- raw.filter(l_freq=0.1, h_freq=None, picks=['eeg', 'eog'])：EEG 和 EOG 都滤
- MNE 默认 FIR：零相位，Hamming 窗，过渡带 0.1 Hz（−6 dB 在 0.05 Hz），核长 33793 点（33.0 s）
- 输出：data/step2_hp-raw.fif
- EEG 通道平均标准差 166.5 → 30.1 µV；直流偏移（通道均值的绝对值）4523 → 0.14 µV
- EEG 功率：0.01–0.05 Hz 31494 → 1392（降 96%）；0.05–0.1 Hz 10613 → 6814（过渡带，部分保留）；1–30 Hz 3.94 → 3.93（不变）
