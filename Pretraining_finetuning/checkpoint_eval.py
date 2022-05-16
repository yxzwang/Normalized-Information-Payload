import os, json, subprocess
PATH = "./bert80kconsine/wikitext-103-raw-v1/"
SAVEFILE = "./bert80kmlm.json"

def run_bash(filepath,taskname):
    result = os.system(
    "CUDA_VISIBLE_DEVICES=0 python run_mlm.py \
    --model_name_or_path {} \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --report_to none \
    --output_dir {} \
    --do_eval \
    --fp16 \
    --fp16_full_eval \
    --per_device_eval_batch_size 8 >&1 | tee {}/{}.txt".format(filepath,filepath,filepath,taskname)
    )
    return result

if __name__ == '__main__':
    # 打开文件
    dirs = os.listdir(PATH)
    # 输出所有文件和文件夹
    taskname = "wikitext103"
    filepath_list = []
    for file in dirs:
        filepath = os.path.join(PATH, file)
        if os.path.isdir(filepath) and "checkpoint" in filepath:
            print("### {} ###".format(filepath))
            run_bash(filepath,taskname)
            filepath_list.append("{}/{}.txt".format(filepath,taskname))

    print("### Finished ###")
    outputs = {}
    for filepath in filepath_list:
        with open(filepath,'r') as f:
            tmp = f.read()
        start = tmp.find("*** Evaluate ***")
        outputs[filepath] = tmp[start:]
    with open(SAVEFILE,'w') as file_obj:
        json.dump(outputs,file_obj)