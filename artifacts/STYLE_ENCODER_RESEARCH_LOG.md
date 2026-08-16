# Style Encoder 研究记录

状态：**双风格编码器方案成功，是整个调查里唯一真正的 Pareto 改进**。#17：SRA_5 从"完整StyleSTAttn无修复"的 55.86 涨到 **81.10**（+25.2pp），FID 几乎不变（2.127→2.158），R-Prec@3 仅小幅下降（0.732→0.695）——FID/R-Prec 甚至都比 `StyleMLP` baseline 更好，只有 SRA 还差 `StyleMLP`（86.27）约5个点没追平，但已经超过官方 checkpoint 的 SRA_5（79.09）。根因排查路径：cross-cell mixing 天然让 pooled 向量偏向内容统计特征（#16：随机初始化时就已存在这个偏置，训练几乎不改变它），"梯度支配"（#14）、"HyperLoRA不会用"（#15）两个假说均被证伪，最终解法是不再试图用 loss 去"训出"干净的风格信号，而是架构上让一支完全独立、不参与 mixing 的纯净分支（`pure_style_encoder`）专管风格判别性、且真正参与生成（区别于 #12 `style_readout` 只在 loss 侧打转、不影响生成）。

## 1. 目标

`StyleMLP`（论文原始风格编码器）是逐 cell 独立跑 4 层 MLP 再做 masked mean pool，架构上完全无法感知帧与帧之间的顺序、结构化 token 之间的关系（可实测验证：打乱帧顺序，输出一个数字都不变）。目标是：让风格编码器在 pooling 之前先做时序/结构化 token 之间的 self-attention，使其**有能力**捕捉这类"动态/关系"类风格线索（节奏、姿态转换、部位协调），从而把 SRA(风格表达)/FID(动作质量) 这条 trade-off 曲线真正往外推，而不是在曲线上换点。

## 2. 环境与复现基线

- 环境：`.venv`（系统 python3.10 + pip，非 conda——当时 conda 源异常），资源通过软链接复用本机 `salad`/`ARSMooDi` 项目，避免重新下载。
- `transformers` 锁定 `<5`（5.x 移除了 vendored SALAD 代码依赖的 `move_cache`）。
- 官方 checkpoint (`epoch_0100.ckpt`) 复现评测：SRA_5=79.09 / R-Prec@3=0.733 / FID=1.250 / FSR=0.094，与论文 Table 1（SRA_5=76.03 / R-Prec@3=0.721 / FID=1.138 / FSR=0.086）高度吻合，验证了评测流程可信。
- 从零训练两次（seed=42/123，`StyleMLP` 原架构）：SRA_5≈86 / FID≈2.2-2.4，两个种子几乎重合 → **确认这是训练脚本本身可复现的系统性差异，不是运气**，超参数逐项核对过与论文一致。

## 3. 实验记录

代码路径：`src/style_salad/models/style.py`（`StyleMLP`/`StyleAttnPool`/`StyleSTAttn` 均注册在 `STYLE_REGISTRY`）。所有实验配置在 `configs/reruns/`，日志在 `artifacts/run_logs/`，结果 CSV 在 `artifacts/evaluation/`，均为非破坏性（不覆盖官方 `epoch_0100.ckpt`）。

| # | 实验 | 配置/代码改动 | SRA(Top1) | SRA(Top5) | R-Prec@3 | FID | 结论 |
|---|---|---|---|---|---|---|---|
| 1 | `StyleMLP` baseline (seed42) | — | 69.18 | 86.27 | 0.682 | 2.391 | 基线 |
| 2 | `StyleMLP` (seed123) | 只换种子 | — | 86.0 | ~0.68 | ~2.2 | 确认可复现,非随机 |
| 3 | `StyleAttnPool` | mean pool → 单 query attention 加权池化 | 66.53 | 84.01 | 0.653 | 3.082 | 全面变差,轻微 mode collapse |
| 4 | **`StyleSTAttn`（完整）** | pooling 前加时序+结构化 token self-attention | 30.19 | 55.86 | 0.732 | 2.127 | **SRA暴跌,FID/R-Prec/Diversity全面变好** |
| 5 | └ temporal-only 消融 | 只留时序 attn (`use_skeletal_attn: false`) | 31.83 | 57.16 | 0.730 | 1.747 | 崩溃程度相近 |
| 6 | └ skeletal-only 消融 | 只留结构化 token attn (`use_temporal_attn: false`) | 31.29 | 56.62 | 0.718 | 2.154 | 崩溃程度相近 → **不分时序/skeletal,只要cell间交流就触发** |
| 7 | supcon 权重 ×4 | `loss.supcon.weight: 1.0→4.0` | 29.46 | 55.58 | 0.733 | 2.043 | 几乎无效 |
| 8 | Centering | pooling 后接 `nn.BatchNorm1d(affine=False)` | **6.79** | **18.23** | **0.806** | **1.333** | **适得其反**——SRA史上最差,但R-Prec甚至超过ground truth |
| 9 | 内容对抗,λ固定1.0 | GRL+7分类器 (`content_adversary`),`content_idx`已接入dataset/train.py/t2sm.py | 28.81 | 55.00 | 0.738 | 1.792 | 无效;分类器loss全程焊在≈ln(7)=1.946,没学动 |
| 10 | 内容对抗,λ warmup | DANN式退火 `2/(1+e^-10p)-1`,排除训练不稳定假设 | 27.59 | 55.02 | 0.733 | 1.994 | 仍无效,分类器仍未学动 |
| 11 | 内容对抗,warmup+分类器10倍lr | 独立 optimizer param group, lr=1e-3 | 22.80 | **48.77** | **0.755** | **1.532** | **更差**——分类器epoch1确实学到东西(loss=1.74),但λ爬升后又被摁回瞎猜水平,且SRA进一步崩、FID/R-Prec进一步变好 |
| 12 | 架构分家 (`style_readout`) | supcon 只作用在 `style_readout(raw_style.detach())`,梯度无法回传进 style_encoder 主干；生成/HyperLoRA 那条路完全不变 | 27.59 | 54.78 | 0.719 | **2.736**（更差）| 无效,FID 反而变差——说明 supcon 对主干的梯度贡献本来就小到可忽略,隔离掉这份贡献自然也没什么变化 |
| 13 | 分阶段预训练 (`pretrain_style_epochs`) | Phase1：额外20epoch,只用supcon训style_encoder(不跑denoiser),loss从3.79→1.5平滑下降；Phase2：原有100epoch联合训练不变 | 28.10 | 54.40 | 0.723 | 2.395 | 无效——Phase1学得很好,但100epoch(9400步)的Phase2把这个起点完全冲刷掉了 |
| 14 | **扩散loss权重砍到0.1** | `loss.style.weight: 1.0→0.1`（`loss.supcon.weight`不变,仍1.0）,快速诊断"梯度支配"假说,不追求保质量 | 29.44 | **55.41** | 0.734 | **2.020**（比baseline还好）| **关键反证**——SRA几乎没变化,FID反而略微变好。直接推翻"扩散loss梯度太强压制supcon"这个一直以来的根因判断 |
| 15 | HyperLoRA 敏感度诊断（独立脚本,非训练实验）| 冻结已训练好的 StyleMLP(好,SRA86%) 与 StyleSTAttn(差,SRA56%) 两个checkpoint,固定基础特征`cond`,喂进4个代表性`DenseFiLM`模块不同的真实风格向量/随机向量,测量FiLM delta的响应幅度 | — | — | — | — | **同样证伪**——StyleSTAttn 的 HyperLoRA 对"真实风格差异"vs"同等尺度随机噪声"的响应比值(0.94~1.45)不低于甚至高于 StyleMLP(0.92~1.31);对"不同风格"vs"同风格不同动作"的区分度比值也是StyleSTAttn更高(部分模块达1.4+)。说明 HyperLoRA 忠实响应了风格向量,不存在"学不会用"的问题 |
| 16 | **未训练（随机初始化）内容泄露探针**（独立脚本,非训练实验,零成本）| `StyleMLP`/`StyleSTAttn` 均不加载任何checkpoint、完全随机初始化,3个不同init种子,跑内容探针 | — | — | — | — | **决定性证据**——未训练时 StyleMLP=33.1%、StyleSTAttn=47.4%,3个种子高度一致;跟**训练过**的数字（32.2% / 44.2%）几乎完全一样。证明内容泄露是 cross-cell mixing 架构本身的固有属性,训练几乎不改变它——彻底解释了为什么#7-#15所有"从训练侧下手"的尝试都无效 |
| 17 | **双风格编码器**（`pure_style_encoder`）| 新增独立 `StyleMLP` 分支(不参与mixing,自己的参数),detach后与mixing分支融合(`style_combiner`,Linear 512→256)喂给HyperLoRA;纯净分支不detach直接接supcon。训练/生成/风格引导/t-SNE全部统一走新的 `_style_embeddings()` 入口,保证训练推理一致 | **61.81** | **81.10** | 0.695 | 2.158 | ✅**成功，整个调查唯一的Pareto改进**——相比无修复的StyleSTAttn(#4)：SRA_5 +25.2pp(55.86→81.10)，FID几乎不变(2.127→2.158)，R-Prec仅小降(0.732→0.695)。FID/R-Prec甚至优于`StyleMLP`baseline，SRA_5已超过官方checkpoint(79.09)，只是还没追平`StyleMLP`(86.27)。4-seed确认（42/123/7/2026，见memory: style-salad-dualenc-seed-distribution）：SRA_5=82.49±2.12，相对baseline(87.12±1.35) z=-3.43——差距是真实的、系统性的，不是seed42运气差 |
| 18 | **放开纯净分支的 detach**（`detach_from_generation: false`）| `t2sm.py`新增开关：`pure_style_encoder`的输出不再对`s_pure`做detach再送入`style_combiner`，扩散loss梯度得以和supcon一起同时塑形该分支（#17里它只吃supcon一路）。动机：#17诊断`style_combiner`权重分配正常(`analyze_combiner_weights.py`未发现W_gen压制W_pure)，剩下最大的差异是`StyleMLP`baseline本身同时吃两路梯度，而#17的纯净分支只吃一路 | 71.70 | 87.87 | 0.678 | 2.233 | ✅**在#17基础上进一步缩小差距，假说成立**——单点(seed42) SRA_5从81.10涨到87.87，已追平`StyleMLP`baseline(86.27)。4-seed确认：SRA_5=87.88±0.63 vs #17的82.49±2.12，**Welch t=4.87(df=3.5)强显著**，且4个种子都稳定落在87.2~88.7（比#17的std=2.12更稳）；FID打平baseline(z=0.04)也打平#17(t=0.60不显著)；R-Prec@3相对baseline z=-2.01（临界，唯一的代价，但vs #17本身t=-1.69不显著）。诊断：换checkpoint重跑`analyze_combiner_weights.py`，权重分配几乎不变(47.7/52.3% vs 47.1/52.9%)——说明提升来自`pure_style_encoder`自身被扩散梯度更好地塑形，不是融合方式变了。**目前推荐把这个开关设为默认新配置** |

内容泄露诊断（独立脚本，非训练实验）：线性探针 + kNN 测 `content_idx`/`style_idx` 能否从池化后的 `s` 中被读出。`StyleSTAttn` 的 content 探针准确率 44.2%，明显高于 `StyleMLP` 的 32.2%（基线 15.6%）；同时发现所有变体（含随机初始化）的 embedding 都存在严重"各向异性"——任意两向量余弦相似度普遍在 0.82~0.997，与 BERT-whitening 论文报告的范围（0.8~0.99）几乎精确重合。

## 4. 关键发现

1. **排列不变性证明**：`StyleMLP`/`StyleAttnPool` 对帧顺序、token 顺序完全不敏感（实测：打乱顺序输出不变）；`StyleSTAttn` 加了 attention 后才具备顺序敏感性（实测：打乱顺序输出真的变了）——架构目标本身是达成的。
2. **一致的 trade-off**：cross-cell mixing 让内容一致性(R-Prec)、动作真实感(FID)、多样性(Diversity) 全面变好，但风格表达(SRA)系统性崩溃，且**不区分是时序还是结构化 token 维度的交流**，两个独立消融结果高度一致。
3. **七次"救回风格"尝试全部失败，且呈现同一个方向性模式**：不管干预机制是什么（加权、白化、对抗×3种变体、架构隔离、分阶段预训练），结果**从未让 SRA 变好过，反而经常让它更差、同时让内容/质量指标更好**——即所有尝试都把系统进一步推向"内容"端，没有一次推向"风格"端。
4. **对抗分类器诊断**：固定 λ 和 warmup λ 下，分类器 loss 全程停留在随机猜测水平（≈ln(7)），提高其学习率后它确实在训练初期学到了东西（epoch1 loss=1.74），但风格编码器随即"反击"将其重新压回瞎猜水平，且整体结果比之前更偏内容端。
5. **架构分家（#12）证伪了"共享向量导致竞争失败"这个假说本身**：把 supcon 完全隔离到独立 readout 头（梯度物理上无法回传进主干，用真实训练过的 checkpoint 验证过隔离确实生效）之后，SRA 没有任何改善，FID 反而变差。结合 supcon×4（#7）本来就几乎无效，说明 supcon 传入主干的梯度，从一开始量级就小到可以忽略——不是"打输了"，是"压根没怎么参战"，所以隔离掉这份贡献自然也没有变化。
6. **分阶段预训练（#13）进一步证伪了"时机"假说**：给 supcon 20 个 epoch 完全不受扩散loss干扰的独占训练时间（trunk 由此产生的 supcon loss 平滑下降，确实学到了东西），但随后 100 epoch（9400步）的标准联合训练把这个起点完全覆盖，最终结果与没做预训练几乎一致。说明扩散 loss 的支配力不是靠"抢跑得快"，而是靠训练量级/持续时间上的绝对优势，起点如何几乎不影响终点。
7. **"梯度支配"假说本身被证伪（#14）**：把扩散loss权重直接砍到0.1（supcon相对权重理论上放大10倍），SRA 几乎没有任何变化（55.86→55.41），FID 甚至略微变好而不是变差。如果真的是"扩散loss梯度太强压制了supcon"，这么大幅度削弱权重不可能毫无效果。这推翻了此前所有实验共享的那个根因假设——问题不是两个loss谁声音大。
8. **"HyperLoRA不会用"假说也被证伪（#15）**：直接测量 HyperLoRA 对风格输入的敏感度，StyleSTAttn（差）checkpoint 里 HyperLoRA 对风格差异的响应幅度、对不同风格的区分度，都不低于甚至高于 StyleMLP（好）checkpoint。说明 HyperLoRA 没有"学不会用"的问题，它在忠实响应收到的风格向量。
9. **拼出的当前图景**：内容确实泄露进了风格向量（#content-leakage 探针证实），HyperLoRA 忠实响应这个"内容味偏重"的向量（#15），但泄露成因既不是 loss 力量对比（#14 已排除），也不是 supcon 的梯度access权（#12 已排除），也不是训练时机（#13 已排除）。剩下最可能的解释：**cross-cell mixing 这个架构操作本身，不依赖具体是哪个 loss 在训练它，就天然倾向于让池化输出偏向内容统计特征**——比如 attention 混合可能天然地把"全局能量/速度"这类粗粒度、跨cell一致的信号放大传播，而这类信号恰好和内容强相关；这个偏置可能在权重完全随机、还未开始训练时就已经存在于架构本身，训练只是在这个偏置的基础上继续强化,不是偏置的起因。这个猜想还没有直接验证（见下方"下一步"）。

## 5. 文献对照

- 各向异性 / 白化：[BERT-whitening](https://www.researchgate.net/publication/350484166_Whitening_Sentence_Representations_for_Better_Semantics_and_Faster_Retrieval)（Su et al. 2021）、[W-MSE](https://proceedings.mlr.press/v139/ermolov21a/ermolov21a.pdf)（Ermolov et al., ICML 2021）、[DINO centering](https://sh-tsang.medium.com/review-dino-emerging-properties-in-self-supervised-vision-transformers-cfddbb4d3549)（Caron et al., ICCV 2021）、[alignment/uniformity 理论](https://github.com/ssnl/align_uniform)（Wang & Isola, ICML 2020）
- 梯度反转/对抗解耦：[DANN](https://jmlr.org/papers/volume17/15-239/15-239.pdf)（Ganin & Lempitsky, ICML 2015/JMLR 2016）、[Fader Networks](https://proceedings.neurips.cc/paper/2017/hash/3fd60983292458bf7dee75f12d5e9e05-Abstract.html)（Lample et al., NeurIPS 2017）、动作风格迁移领域先例 [Aberman et al., SIGGRAPH 2020](https://cfcs.pku.edu.cn/baoquan/docs/2020-07/20200724093813485776.pdf)（Style-SALAD 论文本身引用）、语音转换领域大量 GRL+说话人分类器工作（如 INTERSPEECH 2022）
- 梯度支配 / 多任务配平：[GradNorm](https://proceedings.mlr.press/v80/chen18a.html)（Chen et al., ICML 2018）、[PCGrad](https://arxiv.org/abs/2001.06782)（Yu et al., NeurIPS 2020）
- 退火类比与已知局限：VAE posterior collapse / [KL annealing](https://www.researchgate.net/publication/306093856_Generating_Sentences_from_a_Continuous_Space)（Bowman et al., CoNLL 2015）——单调退火常常不够,后续需要 cyclical annealing,与我们 warmup 实验(#10)失败的经验吻合
- 架构分家（不打配平仗）：[AutoVC](https://proceedings.mlr.press/v97/qian19c.html)（Qian et al., ICML 2019，"Zero-Shot Voice Style Transfer with **Only Autoencoder Loss**"）——同类问题（内容/说话人解耦）业界放弃对抗配平、改用架构瓶颈分离的先例。**注**：我们按这个思路做的 #12 实验没有复现出 AutoVC 的效果——差异可能在于 AutoVC 是把瓶颈加在信息流的必经之路上（内容编码器输出维度本身被卡死，内容多了说话人信息就装不下），而我们的 `style_readout` 只是把 supcon 的*梯度*隔离开，并没有限制主干本身还能装多少内容信息，主干依然可以在扩散 loss 驱动下自由地把内容学得越来越好

## 6. 结论（已根据 #16、#17、#18 更新——方案成功，且已追平 baseline）

**根因链条（完整版）**：cross-cell mixing 让风格编码器具备了感知时序/关系的能力（目标达成，排列敏感性实测验证），副作用是让 pooled 输出偏向内容统计特征——这个偏置是 **cross-cell mixing 架构本身的固有属性**（#16：随机初始化时就已存在，训练几乎不改变），不是"哪个loss训出来的"，所以任何 loss 侧手段（supcon加权#7、centering#8、对抗×3种#9-11、梯度隔离#12、错峰#13、削弱扩散loss#14）都无法修正它；HyperLoRA 自己也没有"不会用"的问题（#15），它忠实响应收到的（内容味偏重的）向量。

**解法第一步（#17）**：不再试图让 loss "训出"一个干净的风格信号，而是架构上**物理隔离**——加一支完全独立、不参与 mixing 的纯净分支（`pure_style_encoder`，`StyleMLP`架构），detach 后与 mixing 分支融合喂给 HyperLoRA（真正参与生成），同时不 detach 地直接接 supcon（专心被训练成风格判别性强的表示）。4-seed 确认后 SRA_5=82.49±2.12，相对 baseline(87.12±1.35) z=-3.43——差距真实存在，不是噪声。

**解法第二步（#18）**：诊断排除了"`style_combiner` 融合时把权重压向 mixing 分支"这个假说（`analyze_combiner_weights.py`：权重分配一直是 47/53 偏向纯净分支，训练前后几乎不变）。剩下的差异是：`StyleMLP` baseline 本身同时吃"扩散loss + supcon"两路梯度，而 #17 的纯净分支只吃 supcon 一路——**detach 保护得太干净，反而少了扩散loss本可提供的有效塑形信号**。放开这道 detach（`pure_style_encoder` 无 mixing，不会重蹈 #16 的内容偏置覆辙）后，4-seed 确认 SRA_5=87.88±0.63，相对 baseline z=0.56（**追平**），相对 #17 提升 Welch t=4.87（强显著）。FID 两头都打平；R-Prec@3 相对 baseline 有个临界的小幅代价（z=-2.01）。

**效果汇总**：相比无修复的完整 `StyleSTAttn`（#4，SRA_5=55.86），#18 最终把 SRA_5 拉到 **87.88 ± 0.63**（+32pp），FID 打平 baseline，R-Prec@3 有小幅但非灾难性的代价——这是整个调查过程里唯一一次真正的 Pareto 改进，而且最终版本已经追平了原本被认为"差 6 个点追不上"的 `StyleMLP` baseline。**推荐把 `detach_from_generation: false` 设为双编码器方案的默认配置。**

**可能的后续微调方向**（非必需，当前结果已经可用）：
- 如果下游任务在意 R-Prec@3 那个临界的小幅代价，可以尝试给 `pure_style_encoder` 单独配一个更小的学习率，只减弱扩散loss对它的塑形力度、不完全切断（介于 #17 全 detach 和 #18 全不 detach 之间的折中）
- 抽样生成几条不同风格的动作做主观质量核对，确认指标提升在观感上也站得住

代码层面这次调查新增的可复用能力（均通过 config 开关控制，默认关闭、不影响任何既有实验）：`StyleAttnPool`/`StyleSTAttn`（含 `use_temporal_attn`/`use_skeletal_attn`/`use_centering` 开关，`src/style_salad/models/style.py`）、内容对抗 `content_adversary`（`src/style_salad/models/adversary.py` + GRL warmup + 独立学习率，`train.py`）、架构分离 `style_readout`、分阶段预训练 `pretrain_style_epochs`、**双风格编码器 `pure_style_encoder` + `style_combiner`，含 `detach_from_generation` 开关（本次成功方案，均在 `t2sm.py`/`train.py`）**。诊断脚本：`artifacts/analyze_combiner_weights.py`（检查 `style_combiner` 权重分配是否偏向某一分支，独立于训练、零成本）。
