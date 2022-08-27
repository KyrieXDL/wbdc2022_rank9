# wbdc2022_rank9
2022微信大数据挑战赛rank9


### 预训练时间
50h + 35h + 36h + 30h
### 数据
预训练无标注和训练数据。
微调：训练数据。
### 预训练模型
使用了 huggingface 上提供的 hfl/chinese-macbert-base 模型。链接为： https://huggingface.co/hfl/chinese-macbert-base
使用了 huggingface 上提供的 hfl/chinese-macbert-large 模型。链接为： https://huggingface.co/hfl/chinese-macbert-large
使用了 huggingface 上提供的 hfl/chinese-roberta-wwm-ext 模型。链接为： https://huggingface.co/hfl/chinese-roberta-wwm-ext

使用了 huggingface 上提供的 clip_vit_base_32 模型。链接为：https://huggingface.co/openai/clip-vit-base-patch32

### 算法描述

#### 双流模型
模型整体结构：
采用非端到端的结构
（1）首先使用开源的clip vit（patch=32）的提取每一帧的特征。
（2）视觉部分使用使用三层的transformer encoder对32帧进行帧间的建模（同时添加[cls] token来学习全局的视觉特征）；文本部分对于title，asr，ocr分别截取前90长度，然后用[sep]拼接后使用开源的mac-bert-base提取文本特征；视觉-文本模态的融合，使用两个6层的transfomer deocder进行融合，一个使用文本作为query，视觉作为k，v，另一个反之，（主要为了避免其中一个模态的缺失）；最后拼接模态融合后的[cls] embedding和融合前的[cls] embedding进行分类。

预训练任务：
预训练阶段使用了mlm（mask token modeling）、mfm（mask frame modeling）、vtc（video text contras）、vtm（video text match）四个任务。mlm采用15%mask比例，mfm采用25%mask 比例，vtc任务参考moco加入了队列来存储动量模型的特征，达到扩大batch的效果；vtc任务的负样本对则利用到了对比学习的到的相似度来挖掘难负样本对。

微调：
微调除了分类任务，也加入了vtc任务以及模型蒸馏（使用动量更新的模型来蒸馏当前模型）。


#### 单流模型
base模型整体结构：
采用非端到端的结构
（1）首先使用开源的clip vit（patch=32）的提取每一帧的特征。
（2）将32帧的视觉特征通过linear层映射到和文本相同的维度大小上；然后文本tokenize之后和视觉特征拼接输入到mac-bert-base；最后对last-hidden-state的embedding进行mean pooling之后进行分类；

large模型整体结构：
（1）首先使用开源的clip vit（patch=32）的提取每一帧的特征。
（2）将32帧的视觉特征通过linear层映射到和文本相同的维度大小上；然后文本tokenize之后输入到mac-bert-large，然后经过large bert的前18层之后和视觉特征拼接起来输入到最后6层；最后对last-hidden-state的embedding进行mean pooling之后进行分类；

预训练任务：
预训练阶段使用mlm、mfm、itm任务；mlm任务和mfm都采用15%的mask 比例，同时itm任务将batch内的一半的图像逆序之后作为负样本对；

微调任务：
微调使用了分类人物和视频文本对比学习任务；

#### 模型融合
2个双流模型和两个单流模型以不同权重进行融合
...

### 训练流程
（1）预训练
（2）在有标注数据上微调
### 测试流程
划分10%的数据作为验证集。取验证集上最好的模型来做测试。
