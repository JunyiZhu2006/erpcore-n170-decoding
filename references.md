# 参考文献

格式：APA 第 7 版。每条后面注明在项目里哪一步、用在什么地方。
（1）–（3）是笔记里原有的三条，编号不变；（4）起是新补的。

## 文献

（1）Bentin, S., Allison, T., Puce, A., Perez, E., & McCarthy, G. (1996). Electrophysiological studies of face perception in humans. *Journal of Cognitive Neuroscience, 8*(6), 551–565. https://doi.org/10.1162/jocn.1996.8.6.551
- 第一次报道 N170：在后颞区（当时叫 T5/T6，大致是现在的 P7/P8）最大，右侧更强
- 用在：Step 3 预期脸−车差异出现在后脑两侧（地形图）

（2）Luck, S. J. (2014). *An introduction to the event-related potential technique* (2nd ed.). MIT Press.
- 讲伪迹剔除和校正的那一章
- 用在：Step 2 的伪迹剔除思路（电压过大、眨眼、眼动）

（3）Kappenman, E. S., Farrens, J. L., Zhang, W., Stewart, A. X., & Luck, S. J. (2021). ERP CORE: An open resource for human event-related potential research. *NeuroImage, 225*, 117465. https://doi.org/10.1016/j.neuroimage.2020.117465
- 数据来源；把 N170 定义为 PO8 上"脸减车"差异波在 110–150 ms 的平均幅值，所有被试总平均是负的，峰值约 131 ms
- 用在：数据（Step 1）、预处理流程和行为剔除标准 200–1000 ms（Step 2）、110–150 ms 窗口和 PO8 电极（Step 3），图上的 N170 窗口阴影（Step 4–8）

（4）Nemrodov, D., Niemeier, M., Patel, A., & Nestor, A. (2018). The neural dynamics of facial identity processing: Insights from EEG-based pattern analysis and image reconstruction. *eNeuro, 5*(1), ENEURO.0358-17.2018. https://doi.org/10.1523/ENEURO.0358-17.2018
- 时间分辨的 EEG 模式分析：同一张图最多 4 个 trial 平均成一条 trace、线性 SVM（C = 1，LIBSVM）、留一交叉验证、置换检验 + FDR；时间窗样本（10 ms × 12 个电极 = 60 维）
- 用在：Step 4 的解码流程（方法 1）、Step 5 的置换 + FDR、Step 8 的时间窗解码（方法 2）
- 注意：它做的是面孔**身份**解码、13 个被试的组水平统计，结果数字不能直接和本项目（脸 vs 车、单个被试）比

（5）Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B (Methodological), 57*(1), 289–300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x
- FDR 和 BH 校正方法本身
- 用在：Step 5–8 的多重比较校正（q = 0.05）

（6）Phipson, B., & Smyth, G. K. (2010). Permutation p-values should never be zero: Calculating exact p-values when permutations are randomly drawn. *Statistical Applications in Genetics and Molecular Biology, 9*(1), Article 39. https://doi.org/10.2202/1544-6115.1585
- 为什么 p = (1 + 零分布中 ≥ 真实值的次数) / (1 + 置换次数)，分子分母都加 1
- 用在：Step 5–8 的 p 值公式

（7）Hyvärinen, A. (1999). Fast and robust fixed-point algorithms for independent component analysis. *IEEE Transactions on Neural Networks, 10*(3), 626–634. https://doi.org/10.1109/72.761722
- FastICA 算法
- 用在：Step 2 的 ICA（`method='fastica'`），去掉眨眼（ICA000）和水平眼动（ICA003）成分

（8）Chang, C.-C., & Lin, C.-J. (2011). LIBSVM: A library for support vector machines. *ACM Transactions on Intelligent Systems and Technology, 2*(3), Article 27. https://doi.org/10.1145/1961189.1961199
- SVM 求解器；sklearn 的 SVC 底层就是它，Nemrodov et al. (2018) 也用它
- 用在：Step 4–5 的 CPU 版 SVM；Step 6 的 GPU 版按它的算法重写

（9）Fan, R.-E., Chen, P.-H., & Lin, C.-J. (2005). Working set selection using second order information for training support vector machines. *Journal of Machine Learning Research, 6*, 1889–1918.
- LIBSVM 的 SMO 里"用二阶信息选两个 α"的方法
- 用在：Step 6 `step6_gpu_lib.py` 的 `smo()`

## 软件

（10）Gramfort, A., Luessi, M., Larson, E., Engemann, D. A., Strohmeier, D., Brodbeck, C., Goj, R., Jas, M., Brooks, T., Parkkonen, L., & Hämäläinen, M. (2013). MEG and EEG data analysis with MNE-Python. *Frontiers in Neuroscience, 7*, 267. https://doi.org/10.3389/fnins.2013.00267
- 本项目用的版本：MNE 1.13.2
- 用在：读数据、滤波、ICA、切 epoch、降采样、作图（Step 1–4），FDR 函数 `mne.stats.fdr_correction`（Step 5–8）

（11）Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- 本项目用的版本：scikit-learn 1.9.1
- 用在：Step 4–5 的 `SVC(kernel='linear')`、`StandardScaler`；Step 6、8 和 sklearn 对答案

（12）Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., … Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In *Advances in Neural Information Processing Systems 32* (pp. 8024–8035).
- 本项目用的版本：PyTorch 2.11.0（CUDA 12.8）
- 用在：Step 6、8 的 GPU 计算（只用来做矩阵运算，SVM 算法是自己按 LIBSVM 写的）

（13）Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Río, J. F., Wiebe, M., Peterson, P., … Oliphant, T. E. (2020). Array programming with NumPy. *Nature, 585*(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2
- 用在：全部步骤的数组计算

（14）hu-neuro-pipeline（版本 0.9.0）[软件]. https://github.com/alexenge/hu-neuro-pipeline
- 用在：Step 1 下载 ERP CORE N170 的数据（`pipeline.datasets.get_erpcore`）
- 待核对：作者和推荐的引用格式，以它的 GitHub 页面或文档为准

## 数据

（15）ERP CORE N170 数据集，Open Science Framework 项目 pfde9. https://osf.io/pfde9/
- 本项目只用了被试 sub-001
- 待核对：数据的许可条款（公开代码和图时按要求署名），以 OSF 页面为准；引用时连同（3）一起引用

## AI 协助

（草稿，请按实际情况修改）本项目的代码由 Claude（Anthropic）协助编写，并按 Claude 的建议做了部分分析设计；笔记由我整理，其中个别原理说明由 Claude 撰写，已在对应位置注明。
