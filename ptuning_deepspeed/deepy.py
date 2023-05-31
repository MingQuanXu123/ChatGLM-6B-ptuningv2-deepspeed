#!/usr/bin/env python
# Copyright (c) 2021, EleutherAI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
os.environ["PATH"]=os.environ["PATH"] + ":/root/.local/conda/envs/deepspeed-chatglm/"
os.environ["WANDB_API_KEY"] = "a1808b0a4e412c5fadde3096f97c9ec1cf8f243a"
os.environ["PATH"]+= ":/root/.local/bin/"
import deepspeed
from deepspeed.launcher.runner import main

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

from megatron.neox_arguments import NeoXArgs
from megatron.utils import get_wandb_api_key


neox_args = NeoXArgs.consume_deepy_args()
deepspeed_main_args = neox_args.get_deepspeed_main_args()

# Extract wandb API key and inject into worker environments
wandb_token = get_wandb_api_key(neox_args=neox_args)
if wandb_token is not None:
    deepspeed.launcher.runner.EXPORT_ENVS.append("WANDB_API_KEY")
    os.environ["WANDB_API_KEY"] = wandb_token

if __name__ == "__main__":
    main(deepspeed_main_args)
