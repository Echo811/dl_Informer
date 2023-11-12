## 2023年11月3日
* 数据处理中的 read_data 基本看懂
* Dataset_ETT_hour类已看懂
* __getitem__、__len__()已看懂
* 20点53分
  * 已完成自定义数据集的预测任务
  * 参数命令位于 scripts/NSE-TATA.sh 中

## 2023年11月12日
* 参考内容
> 时间复杂度：Informer跑起来还是很快的，和后续几个论文中自称更快的模型对比来说，Informer的效率是最高的主要还是得益于它计算Attention的时候放弃了一堆query且有个下采样的过程
> Informer我最开始使用的时候，效果死活很垃圾，之后我把中间的下采样删去了，效果一下子变好了。分析因为我们输入的seq_len不大，本身就是概率稀疏注意力，再不断的下采样，最后用到的query也没几个了，这肯定是有问题的。
> Informer主要可以调的一些参数：factor、seq_len, d_model，其余效果都一般。
* 已读内容
  * model.py
  * embed.py
  * encoder.py
  * decoder.py
  * attn.py