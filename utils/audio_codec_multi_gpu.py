"""
Simple Multi-GPU Audio Processing
Each GPU runs 4 models concurrently


CUDA_VISIBLE_DEVICES=3 python multi_gpu_processing_v2.py
"""

import os
import time
import torch
import json
import torchaudio
import torchaudio.transforms as T
from snac import SNAC
from transformers import AutoTokenizer
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Process, Queue
import argparse
from loguru import logger
from copy import deepcopy

# Set multiprocessing start method for CUDA compatibility
mp.set_start_method('spawn', force=True)


class AudioCodec:
    def __init__(self, code_path="hubertsiuzdak/snac_24khz", 
                 tokenizer_name="Qwen/Qwen2.5-0.5B-Instruct",
                 emotion_tag_path=None, device_id=0):
        
        self.device = torch.device(f"cuda:{device_id}")
        self.code_path = code_path
        self.tokenizer_name = tokenizer_name
        self.emotion_tag_path = emotion_tag_path
        self.codebook_size = 4096
        
        # Load components
        self.emotion_tag_list = self._load_emotion_tag()
        self.audio_codec_model = self._load_audio_codec_model()
        self.tokenizer = self._load_tokenizer()
        
    def _load_audio_codec_model(self):
        print(f"Loading SNAC model on {self.device}")
        model = SNAC.from_pretrained(self.code_path).to(self.device)
        return model
        
    def _load_emotion_tag(self):
        if self.emotion_tag_path and os.path.exists(self.emotion_tag_path):
            with open(self.emotion_tag_path, 'r') as f:
                emotion_tag = json.load(f)
            return list(emotion_tag.values())
        return []

    def _load_tokenizer(self):

        """
        This function is used to load the tokenizer.

        args:
            None

        returns:
            tokenizer: tokenizer
        """
        logger.info("loading tokenizer.............")
        tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)

        print(f"tokenizer vocab size: {len(tokenizer)}")

        vocab_size = len(tokenizer)
        logger.info(f"Vocabulary size: {vocab_size}")
        # Add special tokens
        Start_End_tokens = [
            "<|TEXT_GENERATION_START|>",
            "<|TEXT_GENERATION_END|>",
            "<|TEXT_UNDERSTANDING_START|>",
            "<|TEXT_UNDERSTANDING_END|>",
            "<|SPEECH_GENERATION_START|>",
            "<|SPEECH_GENERATION_END|>",
            "<|SPEECH_UNDERSTANDING_START|>",
            "<|SPEECH_UNDERSTANDING_END|>",
        ]

        emotion_tag_list = self.emotion_tag_list
        # print(f"self.emotion_tag_list: {emotion_tag_list}")
        logger.info(f"emotion_tag_list length: {len(emotion_tag_list)}")

        speech_token_lenght = 7 * self.codebook_size

        logger.info(f"speech_token_lenght: {speech_token_lenght}")

        new_speech_tokens = [f"<|s_{i}|>" for i in range(speech_token_lenght)]
        all_new_tokens = Start_End_tokens + emotion_tag_list + new_speech_tokens
        num_added_tokens = tokenizer.add_tokens(all_new_tokens)

       

        tokenizer.pad_token_id = 128001
        tokenizer.unk_token_id = 128002

        logger.info(f"num_added_tokens: { len(all_new_tokens)}")
        
        logger.info(f"\nAdded {num_added_tokens} special tokens")
        logger.info(f"New vocabulary size: {len(tokenizer)}")
        logger.info(f"Pad token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
        logger.info("tokenizer loaded.............")

        logger.info(f"UNK token: {tokenizer.unk_token}, ID: {tokenizer.unk_token_id}")
        
        return tokenizer


    def remove_duplicate_frames(self, codes_list):
        vals = codes_list
        if len(vals) % 7 != 0:
            raise ValueError("Input list length must be divisible by 7")
        result = vals[:7]
        removed_frames = 0
        for i in range(7, len(vals), 7):
            current_first = vals[i]
            previous_first = result[-7]
            if current_first != previous_first:
                result.extend(vals[i:i+7])
            else:
                removed_frames += 1
        return result

        
    def tokenise_audio(self, codes):
        all_codes = []
        for i in range(codes[0].shape[1]):
            all_codes.append(codes[0][0][i].item())
            all_codes.append(codes[1][0][2*i].item()+4096)
            all_codes.append(codes[2][0][4*i].item()+(2*4096))
            all_codes.append(codes[2][0][(4*i)+1].item()+(3*4096))
            all_codes.append(codes[1][0][(2*i)+1].item()+(4*4096))
            all_codes.append(codes[2][0][(4*i)+2].item()+(5*4096))
            all_codes.append(codes[2][0][(4*i)+3].item()+(6*4096))
        
        all_codes = self.remove_duplicate_frames(all_codes)
        
        return all_codes
        
    def snac_audio_encode(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.to(dtype=torch.float32)
        
        if sample_rate != 24000:
            waveform = T.Resample(orig_freq=sample_rate, new_freq=24000)(waveform)
        waveform = waveform.unsqueeze(0).to(self.device)
        
        with torch.inference_mode():
            codes = self.audio_codec_model.encode(waveform)
        return codes
        
    def get_text_ids(self, text):
        text = f"<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>"
        return self.tokenizer.encode(text, add_special_tokens=False)
        
    def get_speech_ids(self, speech_codes):
        return (
            [self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")]
            + [self.tokenizer.convert_tokens_to_ids(f"<|s_{code}|>") for code in speech_codes]
            + [self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")]
        )
        
    def get_text_and_speech_ids(self, text, audio_codes_list):
        text_ids = self.get_text_ids(text)
        speech_ids = self.get_speech_ids(audio_codes_list)
        unk_id = self.tokenizer.unk_token_id
        speech_ids_status = not any(tid is None or tid == unk_id for tid in speech_ids)
        return text_ids, speech_ids, speech_ids_status


def worker_process(worker_id, gpu_id, task_queue, result_queue, config, worker_progress, worker_tasks):
    """Worker process that processes audio files"""
    
    # Initialize AudioCodec for this worker
    try:
        ac = AudioCodec(
            code_path=config['code_path'],
            tokenizer_name=config['tokenizer_name'],
            emotion_tag_path=config['emotion_tag_path'],
            device_id=gpu_id
        )
        
        logger.info(f"Worker {worker_id} ready on GPU {gpu_id}")
        
        # Track worker's own progress
        files_processed = 0
        start_time = time.time()
        
        # Create a progress bar for this worker
        task_count = worker_tasks[worker_id]
        pbar = tqdm(total=task_count, 
                    desc=f"Worker {worker_id} (GPU {gpu_id})",
                    position=worker_id+1,  # Position the bars below the main bar
                    leave=False)  # Don't leave the bar when done
        
        while True:
            try:
                # Get task from queue
                task = task_queue.get(timeout=5)
                if task is None:  # Stop signal
                    break
                    
                item, input_dir, folder = task
                
                # Process the file
                result = process_file(ac, item, input_dir, folder, config['max_duration'])
                result_queue.put(result)
                
                # Update progress
                files_processed += 1
                worker_progress[worker_id] = files_processed
                pbar.update(1)
                
                # Calculate processing speed
                elapsed = time.time() - start_time
                speed = files_processed / elapsed if elapsed > 0 else 0
                
                # Update progress bar postfix
                pbar.set_postfix_str(f"{speed:.2f} files/s | {result['status']}")
                
                # Log progress
                if result['status'] == 'skipped':
                    logger.info(f"Worker {worker_id}: Skipped {item['audio_file']} - {result['reason']}")
                elif result['status'] == 'error':
                    logger.error(f"Worker {worker_id}: Error processing {item['audio_file']} - {result['error']}")
                    
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                result_queue.put({'status': 'error', 'error': str(e)})
                
        # Close the progress bar
        pbar.close()
                
    except Exception as e:
        logger.error(f"Worker {worker_id} initialization error: {e}")
        
    logger.info(f"Worker {worker_id} finished after processing {files_processed} files")


def process_file(ac, item, input_dir, folder, max_duration):
    """Process a single audio file"""
    try:
        audio_path = item["audio_file"]
        text = item["text"]
        
        # Check duration
        if "duration" in item and item["duration"] > max_duration:
            return {'status': 'skipped', 'reason': 'duration_too_long'}
            
        audio_file_name = os.path.basename(audio_path)
        
        # Set paths
        if folder == "processed_sknahin":
            input_audio = os.path.join(input_dir, item["audio_file"])
            output_json_file = os.path.join(input_dir, folder, "json_snac", 
                                          audio_file_name.replace(".wav", "_snac.json"))
            os.makedirs(os.path.join(input_dir, folder, "json_snac"), exist_ok=True)
        else:
            input_audio = os.path.join(input_dir, folder, "wavs", audio_file_name)
            output_json_file = os.path.join(input_dir, folder, "wavs", audio_file_name.replace(".wav", "_snac.json"))
            # os.makedirs(os.path.join(input_dir, folder, "json_snac"), exist_ok=True)
            
        # Debug info
        # print(f"Processing file: {audio_file_name}")
        # print(f"Input audio path: {input_audio}")
        # print(f"Output JSON path: {output_json_file}")
            
        # Check if file already exists
        if os.path.exists(output_json_file):
            return {'status': 'skipped', 'reason': 'already_exists'}
            
        if not os.path.exists(input_audio):
            print(f"Audio file not found: {input_audio}")
            # Check if the directory exists
            wavs_dir = os.path.dirname(input_audio)
            if not os.path.exists(wavs_dir):
                print(f"Wavs directory not found: {wavs_dir}")
                if os.path.exists(os.path.join(input_dir, folder)):
                    print(f"Contents of {os.path.join(input_dir, folder)}:")
                    print(os.listdir(os.path.join(input_dir, folder)))
            return {'status': 'skipped', 'reason': 'audio_not_found'}
            
        # Process audio
        codec_codes = ac.snac_audio_encode(input_audio)
        audio_codes_list = ac.tokenise_audio(codec_codes)
        text_ids, speech_ids, speech_ids_status = ac.get_text_and_speech_ids(text, audio_codes_list)
        
        # Save result
        new_item = {
            "text": text,
            "audio_file": audio_path,
            "text_ids": text_ids,
            "speech_ids": speech_ids,
            "speech_ids_status": speech_ids_status
        }
        
        os.makedirs(os.path.dirname(output_json_file), exist_ok=True)
        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(new_item, f, indent=4, ensure_ascii=False)
            
        return {'status': 'success', 'output_file': output_json_file}
        
    except Exception as e:
        import traceback
        print(f"Error processing file: {traceback.format_exc()}")
        return {'status': 'error', 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description="Simple Multi-GPU Audio Processing")
    parser.add_argument("--gpus", type=str, default="0", help="GPU IDs to use (e.g., '0,1,2,3')")
    parser.add_argument("--workers_per_gpu", type=int, default=8, help="Number of workers per GPU")
    parser.add_argument("--input_dir", type=str, default="/raid/data/multilangual/ja/emilia_dataset", help="Input directory")
    parser.add_argument("--max_duration", type=float, default=30.0, help="Max audio duration (seconds)")
    parser.add_argument("--json_file", type=str, default="label.json", help="JSON filename to look for")
    parser.add_argument("--folders", type=str, default="JA-B000000", help="Comma-separated list of folders to process")
    
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    
    # Configuration
    config = {
        'code_path': "hubertsiuzdak/snac_24khz",
        'tokenizer_name': "Qwen/Qwen2.5-0.5B-Instruct",
        'emotion_tag_path': "./config/emotion_tags.json",
        'max_duration': args.max_duration
    }
    
    print(f"Starting with {len(gpu_ids)} GPUs, {args.workers_per_gpu} workers each")
    print(f"Total workers: {len(gpu_ids) * args.workers_per_gpu}")
    
    # Check CUDA availability
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        for i in gpu_ids:
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Create queues
    task_queue = Queue()
    result_queue = Queue()
    
    # Create a manager for shared objects
    manager = mp.Manager()
    worker_progress = manager.dict()
    worker_tasks = manager.dict()
    
    # Start workers
    workers = []
    worker_id = 0
    
    for gpu_id in gpu_ids:
        for _ in range(args.workers_per_gpu):
            worker_progress[worker_id] = 0  # Initialize progress counter
            worker_tasks[worker_id] = 0     # Initialize task counter
            p = Process(target=worker_process, args=(worker_id, gpu_id, task_queue, result_queue, config, worker_progress, worker_tasks))
            p.start()
            workers.append(p)
            worker_id += 1
    
    # Load and queue tasks
    input_dir = args.input_dir
    folders_to_process = args.folders.split(',')  # Parse folders from command line
    json_filename = args.json_file
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return
    
    print(f"Contents of input directory {input_dir}:")
    print(os.listdir(input_dir))
    
    total_tasks = 0
    for folder in os.listdir(input_dir):
        # if folder not in folders_to_process:
        #     continue
            
        json_file = os.path.join(input_dir, folder, json_filename)
        if not os.path.exists(json_file):
            print(f"JSON file not found: {json_file}")
            print(f"Available files in {os.path.join(input_dir, folder)}:")
            try:
                files = os.listdir(os.path.join(input_dir, folder))
                for f in files:
                    if f.endswith('.json'):
                        print(f"  - {f}")
            except Exception as e:
                print(f"Error listing directory: {e}")
            continue
            
        # Check for wavs directory
        wavs_dir = os.path.join(input_dir, folder, "wavs")
        # if not os.path.exists(wavs_dir):
        #     print(f"Wavs directory not found: {wavs_dir}")
        #     print(f"Contents of {os.path.join(input_dir, folder)}:")
        #     print(os.listdir(os.path.join(input_dir, folder)))
        #     continue
            
        # print(f"\nProcessing folder: {folder}")
        # try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print(f"Found {len(data)} files in JSON")
        
        # Distribute tasks evenly among workers
        num_workers = len(workers)
        worker_index = 0
        
        for item in tqdm(data, desc=f"Queueing tasks for {folder}"):
            task_queue.put((item, input_dir, folder))
            worker_tasks[worker_index] += 1  # Assign task to this worker
            worker_index = (worker_index + 1) % num_workers  # Round-robin assignment
            total_tasks += 1
                
        # except Exception as e:
        #     print(f"Error loading JSON file {json_file}: {e}")
        #     continue
    
    print(f"\nTotal tasks queued: {total_tasks}")
    print("Tasks per worker:")
    for w_id, count in worker_tasks.items():
        print(f"  Worker {w_id}: {count} tasks")
    
    if total_tasks == 0:
        print("No tasks to process. Exiting.")
        # Stop workers
        for _ in workers:
            task_queue.put(None)
        for p in workers:
            p.join()
        return
    
    # Collect results
    start_time = time.time()
    completed = 0
    success_count = 0
    skip_count = 0
    error_count = 0
    
    # For calculating processing speed
    last_update_time = time.time()
    last_completed = 0
    
    # Main progress bar
    pbar = tqdm(total=total_tasks, desc="Overall progress", position=0)
    
    while completed < total_tasks:
        try:
            result = result_queue.get(timeout=10)
            
            if result['status'] == 'success':
                success_count += 1
            elif result['status'] == 'skipped':
                skip_count += 1
            elif result['status'] == 'error':
                error_count += 1
                
            completed += 1
            
            # Update progress bar with custom postfix info
            current_time = time.time()
            elapsed = current_time - start_time
            if elapsed > 0:
                speed = completed / elapsed
                eta = (total_tasks - completed) / speed if speed > 0 else 0
                
                pbar.set_postfix_str(
                    f"Speed: {speed:.2f} files/s | "
                    f"Success: {success_count} | Skipped: {skip_count} | Errors: {error_count} | "
                    f"Elapsed: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}"
                )
            
            pbar.update(1)
            
            # Print detailed progress every 100 files
            if completed % 100 == 0:
                print(f"\nProgress: {completed}/{total_tasks} ({completed/total_tasks*100:.1f}%)")
                print(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
                print(f"Success: {success_count}, Skipped: {skip_count}, Errors: {error_count}")
                if completed > 0 and elapsed > 0:
                    print(f"Overall speed: {completed/elapsed:.2f} files/second")
                    remaining = (total_tasks - completed) / (completed/elapsed)
                    print(f"Estimated time remaining: {time.strftime('%H:%M:%S', time.gmtime(remaining))}")
                
        except:
            print("Timeout waiting for result")
            break
    
    pbar.close()
    
    # Stop workers
    for _ in workers:
        task_queue.put(None)
        
    for p in workers:
        p.join()
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*50}")
    print("FINAL SUMMARY")
    print(f"{'='*50}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Files processed: {completed}")
    print(f"Success: {success_count}")
    print(f"Skipped: {skip_count}")
    print(f"Errors: {error_count}")
    print(f"Processing speed: {completed/total_time:.2f} files/second")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()