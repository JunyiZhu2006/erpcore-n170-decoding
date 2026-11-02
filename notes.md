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

### 2.3.2 低通 30 Hz
- raw.filter(l_freq=None, h_freq=30, picks=['eeg', 'eog'])
- MNE 默认 FIR：过渡带 7.5 Hz（−6 dB 在 33.75 Hz），核长 451 点（0.44 s）
- 输出：data/step2_lp-raw.fif
- EEG 功率：30 Hz 以下变化 < 0.3%；34–38 Hz −96%；38 Hz 以上 ≈ −100%；60 Hz 单点 2.36 → 0.0002
- EEG 平均标准差 30.1 → 27.1 µV；相邻采样点平均跳变 3.72 → 0.66 µV
- 右侧 F4、F8、FC4、C6、PO8 降 26–31%，高频（多半是肌电）偏多；Oz 只降 0.2%，仍 ~100 µV，噪声在低频，不是肌电

### 检查 Oz（step2_check_oz.py，figs/step2_psd_Oz_O1_O2.png）
- 功率谱：Oz 只在 8 Hz 以下比 O1/O2 高（0.1–1 Hz 约 22 倍，1–4 Hz 约 8.5 倍，4–8 Hz 约 3.4 倍）；α 峰正常，13 Hz 以上和 O1/O2 差不多甚至更低
- 按 1 秒窗看峰峰值：中位数 Oz 48.3 / O1 48.4 / O2 47.8 µV，大部分时间正常
- 问题是间歇性大跳变：23 个 1 秒窗（3%）超过 O1/O2 的 3 倍，最大 1669 µV（78 s），集中在 0–8 s 和 75–81 s
- 结论：不是整条坏道，是偶发的电极跳动

### Oz 标为坏导，重做 2.2 平均参考和 2.3.1/2.3.2 滤波
- 2.1 末尾 raw.info['bads'] = ['Oz']；Oz 留在数据里，之后用邻近电极插值
- 2.2 平均参考只用 29 个好的 EEG 通道（每个时间点之和最大 0.004 µV ≈ 0）；MNE 不对坏导做重参考，Oz 仍是原始参考，反正之后会被插值替换
- 滤波 picks 用 pick_types(eeg=True, eog=True, exclude=[])，Oz 也一起滤（直流 5827 → 0.97 µV；40–100 Hz 功率 0.54 → 0.0023）
- 29 个好通道平均标准差：avgref 163.4 → hp 27.7 → lp 24.5 µV（之前 30 通道版本：166.5 → 30.1 → 27.1）
- 注意：figs/step2_psd_Oz_O1_O2.png 是标坏导之前的数据画的
