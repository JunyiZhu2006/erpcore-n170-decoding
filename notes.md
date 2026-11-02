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

### 2.3.3a ICA 拟合与眼电成分候选
- 2.1 加了电极位置：.set 里没有坐标，用 MNE 模板 colin27_1020（standard_1020 在 MNE 1.14 将被移除），match_case=False 让 FP1/FP2 对上 Fp1/Fp2
- 副本：0.1–30 Hz 数据再做 1 Hz 高通；ICA 只用 29 个好的 EEG 通道（不含 Oz、EOG）
- 拟合时跳过峰峰值 > 500 µV 的 2 秒片段：只跳过 1 段（FP1/FP2），占 0.29%
- fastica，random_state=97；平均参考后秩为 28 → 28 个成分
- 输出：data/step2_ica-ica.fif；图：figs/step2_ica_*.png
- find_bads_eog 候选：ICA000（VEOG r=+0.95，前额对称地形图，时间曲线和眨眼一一对应）→ 眨眼
  ICA003（HEOG r=+0.80，左前正右前负，最大 3 次 HEOG 跳变时 ICA003 同步阶跃，step2_check_ica003.py）→ 水平眼动
- 其余成分和 HEOG/VEOG 的 |r| 都 < 0.1

### 电极位置改用数据集自带的 electrodes.tsv（替换上面的 colin27_1020）
- electrodes.tsv：30 个头皮电极落在半径 85 mm 的理想球面上（Cz = 0, 0, 85），是模板位置，不是逐个被试测量的；3 个眼电电极没有坐标
- coordsystem.json 写的是 ARS，但左侧电极 y 为正，实际轴向是 x 朝鼻、y 朝左耳、z 朝上；转成 MNE head 坐标（x 朝右、y 朝鼻、z 朝上，单位米）：(x, y, z) → (−y, x, z) / 1000
- 检查：FP1 (−26.1, 80.8, −4.0) mm 在左前，F8 在右，Oz (0, −85, −1.8) 在正后
- 和 colin27_1020 比：方向中位差 5.8°，最大 10.7°（P9/P10）
- 地形图头圈用 sphere = (0, 0, 0, 0.085)
- ICA 不受影响：28 个成分，候选仍是 ICA000（VEOG +0.95）、ICA003（HEOG +0.80）；只重画了地形图

### 2.3.3b 去掉 ICA000、ICA003
- 逐个看过 28 个成分后只去掉这两个；ICA010（P9/P10，刺激反应 151 ms 达峰）是 N170 成分，保留；它的脸−车差异在 28 个成分里最大，但方向随窗口变：110–150 ms 为 +22.7、130–200 ms 为 −25.6（单位：刺激前波动的标准差；成分正负号本身没有意义），剔除后再看；肌电（001/007/008/009/020）和扰动（006/018 等）不动，留给 epoch 剔除
- ica.exclude = [0, 3]，ica.apply 作用在 0.1–30 Hz 正式数据上 → data/step2_ica-raw.fif
- 只改 29 个 ICA 通道；EOG 通道（含 HEOG/VEOG）保持校正前，Oz 不变（仍是坏导）
- 眨眼（VEOG 检出 152 次）平均峰峰值：FP1 123.4 → 6.8 µV，FP2 138.5 → 9.8 µV（figs/step2_ica_blinks_before_after.png）
- 与 VEOG 的相关：FP1 −0.85 → −0.03，FP2 −0.92 → −0.29；后部 O1/O2/PO7/PO8 原来有 +0.6~0.7（平均参考把眨眼带到后脑），校正后 ≈ 0
- 与 HEOG 的相关：F7 0.23 → 0.06，F8 −0.33 → 0.08
- 29 个 EEG 平均标准差 24.5 → 17.3 µV
- N170 电极 ERP（全部 trial，未剔除）：−200~500 ms 校正前后 r = 0.97~0.997，平均差 0.3~0.4 µV；500~800 ms 差异较大（1.8~2.6 µV），是刺激后的眨眼被去掉了
- 脸−车（ERP CORE 窗口 110–150 ms）：P7 −1.00 → −1.38，P8 +0.90 → +0.07，PO7 −0.29 → −0.83，PO8 −0.18 → −0.97 µV（figs/step2_ica_erp_before_after.png）
- 注：一开始误用了自选的 130–200 ms，已统一改成 ERP CORE 的 110–150 ms（事先定好的窗口，不按本被试数据挑）

### 2.3.4 Oz 插值 + 30 通道平均参考
- ICA 之后再插值：interpolate_bads(reset_bads=True, origin=(0, 0, 0))，球面样条，用其余 29 个电极
- 再做一次平均参考（30 个通道），每个时间点之和 = 0；插值的 Oz 是其他通道的线性组合，30 个通道的秩是 28
- Oz 标准差：原始 100.0 → 插值后 12.4 µV（O1 13.3，O2 13.1）；插值 Oz 和 O1/O2 平均的相关 0.97
- 输出：data/step2_interp-raw.fif

### 2.4 切 epoch + 基线校正
- 刺激 1–180 共 320 个，按类别打标签 face/car/scrambled_face/scrambled_car + 编号（如 'face/12'）
- tmin −0.2，tmax 0.8，baseline (−0.2, 0)；1024 Hz 下第一个点是 −200.2 ms，基线窗口实际 205 个点（−199.2 ~ 0 ms）
- 保留 30 EEG + 5 EOG（EOG 是 ICA 前的，留给剔除用）；还没剔除任何 epoch
- 每类 80 个；基线均值 max 2e−14 µV（= 0）
- 输出：data/step2_epochs-epo.fif（43.8 MB）；图：figs/step2_epochs_erp_by_category.png
- 110–150 ms 平均（µV，ERP CORE 窗口）：P7 face 1.37 / car 2.67（差 −1.30）；PO7 3.58 / 4.33（差 −0.75）；P8 3.01 / 2.86（差 +0.15）；PO8 3.83 / 4.72（差 −0.89）
- 本被试脸−车差异最负的时刻：P7 171 ms（−5.03），PO7 145 ms（−2.47），P8 149 ms（−3.42），PO8 146 ms（−4.51 µV），大多落在窗口末端，所以 110–150 ms 的平均值偏小；还没剔除，先不下结论
- 给剔除做参考：每个 epoch 最大通道峰峰值 中位 119 / 90% 179 / 99% 280 / 最大 447 µV；VEOG 峰峰值 中位 97 / 90% 318 / 最大 543；HEOG 中位 42 / 90% 79 / 最大 175

### 2.5 剔除 epoch（每个都记下原因）
- 每个 epoch 对所有规则都检查一遍，可以同时有多个原因；明细在 data/step2_reject_log.csv（320 行），被剔除的 epoch 在 step2_clean-epo.fif 的 drop_log 里带着原因；data/step2_epochs-epo.fif 现在带 metadata（全部 320 个）
- 原因代码：EEG_AMPLITUDE 头皮电压过大；BLINK 刺激期间眨眼；SACCADE 刺激期间眼动；NO_RESPONSE 没按键；WRONG 答错（202）；RT_FAST 反应时 < 200 ms；RT_SLOW 反应时 > 1000 ms
- 2.5.1 EEG_AMPLITUDE：每个通道整段 epoch 的峰峰值，同时满足"超过该通道自己的均值 + 3 SD"和"> 100 µV"（各通道实际上限 100–254 µV）→ 13 个；集中在 230、424–428、447、556 s 等头动段
  - 加 100 µV 下限的原因：只用 3 SD 时，安静通道的上限很低（C3 约 45 µV），会剔除最大峰峰值只有 49–100 µV 的正常 epoch（原来 20 个里有 7 个这样）
  - 加下限后：3 个 epoch 被救回（321.5 s、451.7 s、635.3 s），另有 4 个不再算电压问题但仍因答错被剔除
  - log 里 eeg_bad_chs 是触发规则的通道；eeg_max_ptp_ch/eeg_max_ptp_uv 是峰峰值最大的通道，两者可以不同（如 476.9 s：PO4 触发，PO8 最大 112.9 µV 但没超过 PO8 自己的 3 SD）
- 2.5.2a BLINK：刺激呈现的 0–300 ms 内 VEOG（ICA 前）峰峰值 > 本被试眨眼中位大小的一半（连续数据检出 152 次眨眼，中位 341 µV → 阈值 170 µV）→ 2 个（411 s、415 s）；没有一次眨眼峰落在 0–300 ms 内，这 2 个是眨眼的起止边缘
- 2.5.2b SACCADE：0–300 ms 内 HEOG（ICA 前）阶跃（后 200 ms 均值 − 前 200 ms 均值）> 32 µV（约 2°，按约 16 µV/度）→ 10 个；其中 6 个同时答错
- 2.5.3 行为：每个刺激配它后面紧跟的那个反应；答错 74、无反应 2、< 200 ms 4、> 1000 ms 2
- 各类正确率：car 0.69、face 0.79、scrambled_car 0.78、scrambled_face 0.82
- 合计剔除 92 / 320，保留 228：face 60、car 48、scrambled_face 62、scrambled_car 58；只因答错被剔除的有 63 个
- 输出：data/step2_reject_log.csv、data/step2_clean-epo.fif

### ICA 成分编号的保护检查
- 人工确认的眼电成分写在常量 ICA_EOG_CONFIRMED = {0: ('VEOG', 0.95), 3: ('HEOG', 0.80)}：成分编号、它对应的眼电通道、确认时和该通道的 |r|（find_bads_eog 的分数）；ica.exclude 用这个常量的编号
- 检查 1：find_bads_eog 给出的候选和常量里的编号不一致（比如上游多标了坏导、改了滤波，ICA 重跑后编号变了）→ raise RuntimeError 停下
- 检查 2：编号一致，但某个成分和它的眼电通道的 |r| 比确认时下降超过 0.10（ICA_EOG_MAX_R_DROP）→ 也停下；防止"编号碰巧没变、成分已经变了"
- 停下时提示重新看 figs/step2_ica_*.png 再更新常量；报错前 ICA 图已经按新结果重画好；两项检查都在去成分之前，不会写出 step2_ica-raw.fif
- 测试：编号改成 {0, 4} → 检查 1 报错；ICA003 的确认值改成 0.95（现在 0.80，降 0.15）→ 检查 2 报错；原样运行正常跑完，data/ 里 10 个 step2_* 文件和备份逐个 SHA256 一致

## Step 3：脸 vs 车的 ERP
- 脚本：step3_erp.py

### 3.1 分成两堆
- 从 data/step2_clean-epo.fif 取 epochs['face']（60 个）和 epochs['car']（48 个），只留 30 个 EEG 通道；打乱的图不用

### 3.2 每堆取平均
- 每个电极上把同一类的 trial 逐点平均 → 输出 data/step3_face_car-ave.fif（两个 Evoked：face、car）
- 平均压噪声：基线段（−200~0 ms）标准差，30 个电极平均：face 单个 trial 8.19 → 平均后 1.26 µV（缩小 6.5 倍，√60 = 7.7）；car 8.31 → 1.50 µV（5.5 倍，√48 = 6.9）；比 √n 略少，说明噪声不完全是每次独立的随机噪声
- 图：figs/step3_erp_face_car.png（6 个枕颞电极，细线是单个 trial，粗线是平均）；figs/step3_erp_topo.png（30 个电极按头皮位置排）
- （第一版在这里提前算了 6 个电极的脸−车，超出了这一步的要求，已删掉；差异波放到 3.3、3.4）

### 3.3 PO7、PO8 上画脸、车、脸−车
- 差异波 = 脸的平均 − 车的平均（mne.combine_evoked，权重 [1, −1]）
- 图：figs/step3_diff_PO7_PO8.png（红 = 脸，蓝 = 车，黑 = 脸−车，黄色带 = 110–150 ms）

### 3.4 110–150 ms 的平均值 + 误差范围
- 误差范围：两组独立 trial 的差的 95% 置信区间，脸−车 ± 1.96 × SE，SE = √(var_脸/n_脸 + var_车/n_车)；差异波逐点画成灰色阴影
- 一个数：先算每个 trial 在 110–150 ms 的平均，再用同样的公式算脸−车和 95% CI
- PO7：−1.65 µV，95% CI [−4.93, 1.63]
- PO8：+0.37 µV，95% CI [−2.80, 3.53]
- 两个电极的置信区间都跨过 0：这个被试在 110–150 ms 没有可靠的"脸比车低"；PO7 方向对但不显著，PO8 平均接近 0（窗口内差异波从约 +6 摆到约 −7 µV）
- 窗口 110–150 ms 事先定好，不因结果改
