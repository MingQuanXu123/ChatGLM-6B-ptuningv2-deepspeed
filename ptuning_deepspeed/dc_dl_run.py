#!/opt/conda/envs/deep_learning/bin/env python
import getopt
import  json
import os
import sys
import uuid


class DCLogger():
    def info(self, *values, **options):
        self._print(sys.stdout, *values, **options)

    def error(self, *values, **options):
        self._print(sys.stderr, *values, **options)

    def _print(self, out_file, *values, **options):
        sep = options.get("sep", " ")
        end = options.get("end", "\n")
        msg = sep.join(str(x) for x in values) + end
        out_file.write(msg)

def is_python3():
    _PYVersion = sys.version_info[0] + (sys.version_info[1] * 0.1)
    _PY3 = (_PYVersion >= 3.0)
    return _PY3

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def get_args_func(argv):
    name = ''
    file = ''
    queue_time = '0'
    nohup = 'false'
    try:
        opts, args = getopt.getopt(argv, "n:f:q:n", ["name=","file=","queue_time=","nohup="])
    except getopt.GetoptError as e:
        raise e
        # sys.exit(2)
    for opt, arg in opts:  # 依次获取列表中的元组项
        if opt in ("--name"):
            name = arg
        if opt in ("--file"):
            file = arg
        if opt in ("--queue_time"):
            queue_time = arg
        if opt in ("--nohup"):
            nohup = arg
        if opt in ("-n"):
            name = arg
        if opt in ("-f"):
            file = arg
        if opt in ("-q"):
            queue_time = arg
        if opt in ("-n"):
            nohup = arg
    return (name,file,queue_time,nohup)

def analyze_config_params():
    dc_dl_config=os.getenv('DC_DL_CONFIG', '')
    if dc_dl_config == "":
        raise Exception("No environment variables DC_DL_CONFIG")
    else:
        print("=========dc_dl_config====", dc_dl_config)
        data=json.loads(dc_dl_config)
        framework = data['framework']
        if framework is None:
            raise Exception("paramer framework in environment variables DC_DL_CONFIG is not found")
        else:
            if framework.lower() != 'pytorch' and framework.lower() != 'tensorflow2' and framework.lower() != 'oneflow':
                raise Exception("paramer framework in environment variables DC_DL_CONFIG only supports pytorch or tensorFlow2 or oneflow")
            if framework.lower() == 'oneflow':
                framework='pytorch'
            if framework.lower() == 'pytorch':
                framework='pytorch'
            elif framework.lower() == 'tensorflow2':
                framework='tensorflow2'
        worker = data['worker']
        workerResource = {}
        paramResource = worker
        if "cpus" not in paramResource:
            workerResource['cpus'] = 1
        else:
            if is_number(paramResource['cpus']):
                workerResource['cpus'] = int(float(paramResource['cpus']))
            else:
                raise Exception("paramer cpus of worker in environment variables DC_DL_CONFIG is not number")
        if "mem" not in paramResource:
            workerResource['mem'] = 1
        else:
            if is_number(paramResource['mem']):
                workerResource['mem'] = int(float(paramResource['mem']))
            else:
                raise Exception("paramer mem of worker in environment variables DC_DL_CONFIG  is not number")
        if "gpus" not in paramResource:
            workerResource['gpus'] = 0
        else:
            if is_number(paramResource['gpus']):
                workerResource['gpus'] = int(float(paramResource['gpus']))
            else:
                raise Exception("paramer gpus of worker in environment variables DC_DL_CONFIG  is not number")
        if "count" not in paramResource:
            workerResource['count'] = 1
        else:
            if is_number(paramResource['count']):
                workerResource['count'] = int(float(paramResource['count']))
            else:
                raise Exception("paramer count of worker in environment variables DC_DL_CONFIG  is not number")
    return (framework, workerResource)

if __name__ == '__main__':
    if is_python3() == False:
        raise Exception("deepLearning only supported python3")
    sys.path.append("/opt/pylib/datacanvas.zip")
    kernel_id = str(uuid.uuid1())
    (name,file_path,queue_time,nohup)=get_args_func(sys.argv[1:])
    if name == "":
        raise Exception("No deepLearning name")
    if file_path == "":
        raise Exception("No deepLearning runtime file")
    if queue_time == "":
        queue_time = "0"
    (framework, workerResource)=analyze_config_params()
    if os.path.exists(file_path) == False :
        raise Exception("deepLearning runtime file is not found")
    if os.path.isfile(file_path) == False :
        raise Exception("deepLearning runtime file is not file")
    codes = []
    # with open(file_path, "r") as f:  # 打开文件
    #     data = f.read()
    #     codes.append(data)
    # if len(codes) == 0 :
    #     raise Exception("deepLearning is not found code ")
    # print("os.system('{0} -u {1}')".format(sys.executable, os.path.abspath(file_path)))
    codes.append("import os\nresult=os.system('{0} -u {1}')\nif result != 0 : \n    raise  Exception('exec failed')".format(sys.executable, os.path.abspath(file_path)))
    # codes.append("import subprocess\ncommand=['{0}','-u','{1}']\nret = subprocess.run(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding='utf-8',timeout=1)\nif ret.returncode != 0:\n    raise  Exception('exec failed')".format(sys.executable, os.path.abspath(file_path)))

    from datacanvas.util import notebook,dlds
    logger=DCLogger()
    context = {}
    (user_id, instance_id) = notebook.get_user_info()
    context['userId'] = user_id
    context['instanceId'] = instance_id
    context['ipynbName'] = "Terminal-"+kernel_id
    context['kernelId'] = kernel_id
    context['workdir'] = os.getcwd()
    context['pythonPath'] = sys.executable
    resource = {'framework': framework,
                "resourceTypeSettingItems": [{"type": "worker", "userSetting": workerResource}]}

    try:
        (id,unique_name) = dlds.start(name, nohup, context, resource, codes,queue_time)
        logger.info("task unique name is {0}".format(unique_name))
        def query_status(id):
            return dlds.keep_alive(id)
        # notebook.output_log(id,logger,workerResource['count'],0,query_status)
        notebook.output_log_py(id,logger,workerResource['count'],query_status)
    except Exception as e:
        dlds.register(context['ipynbName'])
        raise e
