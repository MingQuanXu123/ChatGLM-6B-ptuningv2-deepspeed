export DC_DL_CONFIG='{"framework":"pytorch","worker": {"count":2,"cpus":4,"mem":8,"gpus":2}}'
python dc_dl_run.py --name=chatglm-task --file=ds_run.py --queue_time=10 --nohup true
