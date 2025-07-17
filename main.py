import sys
import util
import os
import subprocess
from dataset import get_dataset
from function import query_calibrated_answer_probs


def main():
    args, config, mllm_config = util.parse_args_and_configs()

    # logic for local MLLM service
    if args.service == "local_mllm":
        assert mllm_config.local_mllm.use_vllm_serve
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = mllm_config.local_mllm.cuda_devices
        env["XDG_CACHE_HOME"] = mllm_config.local_mllm.cache_dir
        # run vllm serve
        command = [
            "vllm", "serve", mllm_config.local_mllm.model_path,
            f"--gpu_memory_utilization={mllm_config.local_mllm.gpu_memory_utilization}",
            f"--tensor_parallel_size={mllm_config.local_mllm.tensor_parallel_size}",
            f"--host={mllm_config.local_mllm.host}",
            f"--port={mllm_config.local_mllm.port}",
            f"--dtype={args.vllm_dtype}",
            f'--allowed-local-media-path={os.path.join(config.data.data_dir, "images/")}'
        ]
        util.info('main.py', f"Launching vLLM with command: {' '.join(command)}")
        vllm_proc = subprocess.Popen(command,
                                     env=env,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL,
                                     start_new_session=True)
        util.info('main.py', 'Waiting for vLLM server to be ready...')
        util.wait_until_ready(host=mllm_config.local_mllm.host, port=mllm_config.local_mllm.port, subproc=vllm_proc)
        util.info('main.py', 'vLLM server is ready.')

    # run the main script
    # load misc
    get_image_url = util.get_get_image_url_func(service=args.service)
    mllm_class = util.get_mllm_class(service=args.service)
    use_mllm = mllm_class(args=args, config=config)
    prompt_class = util.get_prompt_class(dataset=config.data.dataset)
    use_prompt = prompt_class(get_image_url=get_image_url,
                              parsing_namespace=config.parsing,
                              exemplar_data_points=get_dataset(config=config, split_name="train"))
    data_points = get_dataset(config=config, split_name="test")
    util.info('main.py', f"Number of total data points in {config.data.dataset}: {len(data_points)}")

    # start inference
    log_file_name = "log.json"
    remaining_data_points = util.get_remaining_data_points(
        log_file_path=os.path.join(args.log_save_folder, log_file_name),
        all_data_points=data_points)
    util.info('main.py', f"Number of data points for inference: {len(remaining_data_points)}")
    for data_point in remaining_data_points:
        answer, prob_dict = query_calibrated_answer_probs(data_point=data_point,
                                                          use_prompt=use_prompt,
                                                          use_mllm=use_mllm,
                                                          cared_attribute=config.parsing.cared_attribute,
                                                          n=config.sampling.n,
                                                          temperature=config.sampling.temperature,
                                                          top_logprobs=config.sampling.top_logprobs,
                                                          max_completion_tokens=config.sampling.max_completion_tokens)
        log_data = {"image_id": data_point['image_id'],
                    "answer": answer,
                    "prob_dict": prob_dict}
        util.append_log_and_save(log_file_path=os.path.join(args.log_save_folder, log_file_name),
                                 log_data=log_data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
