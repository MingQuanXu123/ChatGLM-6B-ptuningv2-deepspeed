import os
import time
import torch

slots=str(torch.cuda.device_count())
workers=os.environ.get("ALL_WORKERS")
hosts=workers.replace('"', '').split(",")
print(hosts)
#print("host1=", hosts[1])
hostfiles=""
for host in hosts:
    host=host.strip('\"')
    hostfiles += host + " slots=" + slots  +  "\n"
print(hostfiles)
fileName='hostfilexr'
with open(fileName,'w')as file:
    file.write(hostfiles)   

#cmd="export DLTS_HOSTFILE=hostfilexr ; /opt/conda/envs/deepspeed-dl/bin/python ./deepy.py train.py ./configs/1-3B.yml ./configs/local_setup.yml"
#cmd="/opt/conda/envs/deepspeed-dl/bin/deepspeed --hostfile=hostfilexr main.py \
cmd="/root/.local/conda/envs/deepspeed-chatglm/bin/deepspeed --hostfile=hostfilexr main.py \
      --deepspeed deepspeed.json \
      --do_train \
      --train_file MultiTask_dataset/train_v3.json \
      --test_file MultiTask_dataset/dev_v3.json \
      --prompt_column instruction \
      --response_column output \
      --overwrite_cache \
      --model_name_or_path THUDM/chatglm-6b \
      --output_dir ./output/multitask-chatglm-6b-ft-2e-2 \
      --overwrite_output_dir \
      --max_source_length 200 \
      --max_target_length 200 \
      --per_device_train_batch_size 16 \
      --per_device_eval_batch_size 1 \
      --gradient_accumulation_steps 1 \
      --predict_with_generate \
      --max_steps 7500 \
      --logging_steps 10 \
      --save_steps 1500 \
      --learning_rate 2e-2 \
      --pre_seq_len 128 \
      --quantization_bit 4"



print(cmd)
rank=int(os.environ['RANK'])
print("=============rank==============", rank)
if rank==0:
    os.system(cmd)



