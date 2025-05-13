import json, glob, os
import random, tqdm, re
import numpy as np
import librosa
from transformers import AutoTokenizer
from loguru import logger

# import matplotlib.pyplot as plt

random.seed(37)
_whitespace_re = re.compile(r"\s+")

def get_duration(file_path):
    return librosa.get_duration(path=file_path)


class CodecManagerSaving:
    def __init__(self, tokenizer_name, output_dir):
        self.tokenizer_name = tokenizer_name
        
        self.training_percentages = 0.9
        self.output_dir = output_dir
        self.max_length = 1024
        self.sampling_rate = 24000
        self.language = "bn"
        self.codebook_size = 4096
        self.emotion_tag_path = "/media/sayan/hdd/TTS/QweenTTS/config/emotion_tag.json"
        self.emotion_tag_list = self._load_emotion_tag()
        self.tokenizer = self._load_tokenizer()

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

       

        tokenizer.pad_token_id = 151668
        tokenizer.unk_token_id = 151668

        logger.info(f"num_added_tokens: { len(all_new_tokens)}")
        
        logger.info(f"\nAdded {num_added_tokens} special tokens")
        logger.info(f"New vocabulary size: {len(tokenizer)}")
        logger.info(f"Pad token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
        logger.info("tokenizer loaded.............")

        logger.info(f"UNK token: {tokenizer.unk_token}, ID: {tokenizer.unk_token_id}")
        
        return tokenizer
    
    def _load_emotion_tag(self):
        emotion_tag = json.load(open(self.emotion_tag_path))
        emotion_tag_list = []
        for key, value in emotion_tag.items():
            emotion_tag_list.append(value)
        print(emotion_tag_list)
        return emotion_tag_list

    def nmp_and_memmap(
        self, final_data, memmap_path, shape_path, split_type="None", max_length=1024
    ):

        if final_data:  # Only save if we have sequences
            # Save to disk
            arr = np.memmap(
                memmap_path,
                dtype=np.int32,
                mode="w+",
                shape=(len(final_data), max_length),
            )


            arr[:] = np.array(final_data, dtype=np.int32)

            arr.flush()
            np.save(shape_path, np.array([len(final_data), max_length]))

            print(f"\n=== {split_type} Split Summary ===")
            print(f"Saved {len(final_data)} sequences of length {max_length}")
            print(f"Memmap file size: {os.path.getsize(memmap_path)/1e6:.2f}MB")
            print(f"Shape: {np.load(shape_path)}")
        else:
            print(f"\nWarning: No valid sequences found for split {split_type}")

    def data_compailation(self, path_dir, folder_list, language, max_length=1024, debug=False):
        total_duration = 0
        final_train_data, final_eval_data = [], []
        folders = os.listdir(path_dir)
        total_discard_data, sequence_not_match = 0, 0
        for folder in folders:
            print("folder : ", folder)
            # if folder not in folder_list:
            #     continue

            root_folder = f"{path_dir}/{folder}"

            json_files = glob.glob(root_folder+"/*_snac.json", recursive=True)

            for json_file in json_files:
                print("Processing: ", json_file)
                data = json.load(open(json_file, "r"))
                all_sequences = []
                internal_duration = 0
                for idx, d in enumerate(tqdm.tqdm(data)):

                    if "text_ids" in d and "speech_ids" in d:
                        text_ids, speech_ids = d["text_ids"], d["speech_ids"]
                        # if "duration" not in d:

                        if "duration" not in d or d["duration"]== -1:
                            # print("duration not found")
                            # print("audio_file : ", d["audio_file"])
                            duration = get_duration(os.path.join(path_dir, d["audio_file"]))
                            d["duration"] = duration
                        # if d["duration"] < 0.1:
                        #     total_discard_data += 1
                        #     print(f"Warning: Duration too short ({d['duration']}) for sample {idx}, skipping...")
                        #     continue

                        internal_duration += d["duration"]
                            
                        MAX_TEXT_SPACE = max_length - len(speech_ids)
                        if MAX_TEXT_SPACE < 0:
                            total_discard_data += 1
                            print(
                                f"Warning: Speech sequence too long ({len(speech_ids)} tokens) for sample {idx}, skipping..."
                            )
                            continue

                        # Truncate text to fit
                        truncated_text = text_ids[:MAX_TEXT_SPACE]

                        if debug and idx == 0:
                            print(
                                f"\nTruncated text tokens: {len(truncated_text)} (max available: {MAX_TEXT_SPACE})"
                            )

                        # Build final sequence
                        final_sequence = (
                            truncated_text
                            + speech_ids
                            + [self.tokenizer.pad_token_id]
                            * (max_length - len(truncated_text) - len(speech_ids))
                        )[:max_length]

                        # print(f"Final sequence length: {len(final_sequence)}")
                        if len(final_sequence) > max_length:
                            sequence_not_match += 1
                            print(
                                f"Warning: Final sequence length ({len(final_sequence)}) does not match max_length ({max_length}), skipping..."
                            )
                            continue

                        all_sequences.append(final_sequence)

                if all_sequences:
                    train_data = all_sequences[: int(len(all_sequences) * self.training_percentages)]
                    eval_data = all_sequences[int(len(all_sequences) * self.training_percentages):]

                    print(f"train_data: {len(train_data)}")
                    print(f"eval_data: {len(eval_data)}")

                    print(f" Duration  : {internal_duration/3600} hours")

                    print("Final sequence length number of not match: ", sequence_not_match)

                    total_duration += internal_duration

                    final_train_data.extend(train_data)
                    final_eval_data.extend(eval_data)

        print("Total Final Train Data: ", len(final_train_data))
        print("Total Final Eval Data: ", len(final_eval_data))

        print(f"Total Duration {total_duration/3600} hours")

        print(f"Total Discarded Data: {total_discard_data}")

        # memmap_path = os.path.join(output_dir, f"{split}_input_ids.memmap")
        # shape_path = os.path.join(output_dir, f"{split}_input_ids_shape.npy")

        train_memmap_path = f"{self.output_dir}/train_input_ids.memmap"
        train_shape_path = f"{self.output_dir}/train_input_ids_shape.npy"

        val_memmap_path = f"{self.output_dir}/val_input_ids.memmap"
        val_shape_path = f"{self.output_dir}/val_input_ids_shape.npy"

        test_memmap_path = f"{self.output_dir}/test_input_ids.memmap"
        test_shape_path = f"{self.output_dir}/test_input_ids_shape.npy"

        self.nmp_and_memmap(
            final_train_data, train_memmap_path, train_shape_path, split_type="train"
        )

        print("Train Save process finished")

        self.nmp_and_memmap(
            final_eval_data, val_memmap_path, val_shape_path, split_type="val"
        )
        print("Val Save process finished")
        self.nmp_and_memmap(
            final_eval_data, test_memmap_path, test_shape_path, split_type="test"
        )
        print("Val Save process finished")


if __name__ == "__main__":

    data_path = "/media/sayan/hdd/TTS/BanglaXTTS-dev-saiful/xtts_training/dataset/bn/HQ-STUDIO-TTS-7.16H/dataset"
    output_path = "dataset"
    language = "bn"
    os.makedirs(output_path, exist_ok=True)

    folder_list = []

    CMS = CodecManagerSaving(
        tokenizer_name="Qwen/Qwen2.5-0.5B-Instruct", 
        output_dir=output_path
    )
    CMS.data_compailation(data_path, folder_list, language="bn")
