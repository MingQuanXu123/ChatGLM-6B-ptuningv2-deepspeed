# ChatGLM-6B-PT
本仓库实现了对于 ChatGLM-6B 模型基于 [P-Tuning v2](https://github.com/THUDM/P-tuning-v2) 的微调。在九天平台上利用多机多卡实现了deepspeed分布式训练，可参考"九天平台分布式训练原理与实践V20230317.pdf"文件查看九天分布式训练的说明。

下面对主要的使用数据，代码和环境简要说明。


## 软件依赖
运行微调需要4.27.1版本的`transformers`。除 ChatGLM-6B 的依赖之外，还需要安装以下依赖，具体的环境依赖可以参考本目录的requirements.txt文件
```
pip install rouge_chinese nltk jieba datasets deepspeed==0.8.3
```
## 使用方法

### 数据集
多任务数据集任务为根据指令（instruction）和输入（content）生成输出（output）。主要包括以下六类数据：

业务关键词：2567条

10000相似度： 10257条

业务知识： 34480条

alpaca-zh： 20380条

意图分类： 10002条

闲聊数据集： 30602条。共计约10W条，指令数据集的格式参考alpaca数据集。

```json
{
    "instruction": "为这个问题提供合理的解释。"
    "input": "和包券购买每天惠商城商品 能在和包支付平台绑定我原来每天惠商城商城的帐号吗？",
    "output": "您好！和包帐号对应的手机号将自动关联该手机号对应的每天惠商城商城登录。"
}
```

### 训练

#### P-Tuning v2

运行以下指令进行训练：
```shell
bash start.sh
```
`start.sh` 中的 第一行表示分布式环境配置信息。

任务定义指令，支持以下参数：

-framework，运行框架，取值：tensorflow2/pytorch/oneflow；如需使用deepspeed框架，该参数需设置为pytorch。

-worker，资源配置：count=2,cpus=4,mem=8,gpus=2 分别代表 worker个数,cpu核数,内存大小,gpu个数。

第二行中， --file=ds_run.py 定义了函数的入口文件，修改ds_run.py中的"cmd"命令行，即可定义自己的模型训练函数。

### 模型加载

如果在线加载模型有问题，也可以从本地加载模型，将 `cmd` 中的 `THUDM/chatglm-6b` 改为你本地的模型路径。


### 推理

在 P-tuning v2 训练时模型只保存 PrefixEncoder 部分的参数，所以在推理时需要同时加载原 ChatGLM-6B 模型以及 PrefixEncoder 的权重，因此需要指定 `evaluate.sh` 中的参数：

```shell
--model_name_or_path THUDM/chatglm-6b
--ptuning_checkpoint $CHECKPOINT_PATH
```

仍然兼容旧版全参保存的 Checkpoint，只需要跟之前一样设定 `model_name_or_path`：

```shell
--model_name_or_path $CHECKPOINT_PATH
```

评测指标为中文 Rouge score 和 BLEU-4。生成的结果保存在
`./output/***-chatglm-6b-pt-8-1e-2/generated_predictions.txt`。

