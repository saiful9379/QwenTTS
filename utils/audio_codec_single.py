"""
expose the functions to be used in the inference.py file
CUDA_VISIBLE_DEVICES=0 python data_processing_v2.py
"""

import os
import time
import torch
import json
import torchaudio
import torchaudio.transforms as T
from snac import SNAC
from transformers import AutoTokenizer
from loguru import logger
from tqdm import tqdm
from copy import deepcopy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AudioCodec:
    def __init__(
        self, 
        code_path="hubertsiuzdak/snac_24khz", 
        tokenizer_name="Qwen/Qwen2.5-0.5B-Instruct",
        emotion_tag_path = None, 
        codebook_size=4096,
        decode_status=False,
        encode_status=True,
        saving_pt_file=False,
        ):

        """
        This function is used to initialize the AudioCodec class.

        args:
            None
        """
        self.device = device
        self.code_path = code_path
        self.tokenizer_name = tokenizer_name
        self.codebook_size = codebook_size
        self.decode_status = decode_status
        self.encode_status = encode_status
        self.saving_pt_file = saving_pt_file
        self.emotion_tag_path = emotion_tag_path
        self.emotion_tag_list = self._load_emotion_tag()
        self.audio_codec_model = self._load_audio_codec_model()
        self.tokenizer = self._load_tokenizer()
        
    def _load_audio_codec_model(self):
        audio_codec_model = SNAC.from_pretrained(self.code_path).to(device)
        return audio_codec_model


    def _load_emotion_tag(self):
        emotion_tag = json.load(open(self.emotion_tag_path))
        emotion_tag_list = []
        for key, value in emotion_tag.items():
            emotion_tag_list.append(value)
        print(emotion_tag_list)
        return emotion_tag_list

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



    def load_json(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data


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



    def get_decode_snac_format_codes(self, tokenized_codes):
        """
        Reconstruct SNAC multi-scale codes from tokenized codes.
        
        Args:
            tokenized_codes (list[int]): tokenized and cleaned SNAC codes

        Returns:
            reconstructed_codes (list[Tensor]): original multi-scale SNAC codes ([1,T], [1,2T], [1,4T])
        """

        codebook_size = 4096
        num_levels = 7

        assert len(tokenized_codes) % num_levels == 0, \
            "Tokenized codes length must be divisible by 7."

        T = len(tokenized_codes) // num_levels

        # Prepare empty tensors for each level
        level0 = torch.zeros((1, T), dtype=torch.long, device=self.device)
        level1 = torch.zeros((1, 2 * T), dtype=torch.long, device=self.device)
        level2 = torch.zeros((1, 4 * T), dtype=torch.long, device=self.device)

        for t in range(T):
            idx = t * num_levels
            level0[0, t] = tokenized_codes[idx]
            level1[0, 2 * t] = tokenized_codes[idx + 1] - (codebook_size)
            level2[0, 4 * t] = tokenized_codes[idx + 2] - ( 2 * codebook_size)
            level2[0, 4 * t + 1] = tokenized_codes[idx + 3] - (3 * codebook_size)
            level1[0, 2 * t + 1] = tokenized_codes[idx + 4] - (4 * codebook_size)
            level2[0, 4 * t + 2] = tokenized_codes[idx + 5] - (5 * codebook_size)
            level2[0, 4 * t + 3] = tokenized_codes[idx + 6] - (6 * codebook_size)

        return [level0, level1, level2]

    # def tokenise_audio(self, waveform):
    def tokenise_audio(self, codes):
        all_codes = []
        # # print(f"codes shape: {codes}")
        # print(f"[+] Encoded SNAC codes (lengths): {[c.shape[-1] for c in codes]}")
        # print(f"codes[0].shape[1]: {codes[0].shape[1]}")
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

    # ====== Core Codec Function ======
    def snac_audio_encode(self, audio_path):

        """
        This function is used to encode the audio into SNAC codes.

        args:
            audio_path: path to the audio file
            path_to_output_audio: path to the output audio file

        returns:
            interleaved: interleaved SNAC codes example: codes tensor - [3, 6, 12] to [1, 12]
        
        
        """
        # print(f"[+] Processing: {audio_path}")

        waveform, sample_rate = torchaudio.load(audio_path)
        waveform = waveform.to(dtype=torch.float32)

        # Resample to 24kHz
        if sample_rate != 24000:
            # print(f"[!] Resampling from {sample_rate} → 24000 Hz")
            waveform = T.Resample(orig_freq=sample_rate, new_freq=24000)(waveform)
        waveform = waveform.unsqueeze(0).to(device)  # shape: [1, 1, samples]

        # ===== Encode =====
        with torch.inference_mode():
            codes = self.audio_codec_model.encode(waveform)  # list of [1, T_i]
        return codes


    def snac_audio_decode(self, audio_codec, output_audio):

        """
        This function is used to decode the interleaved SNAC codes into audio.

        args:
            interleaved: interleaved SNAC codes example: codes tensor - [3, 6, 12] to [1, 12]
            path_to_output_audio: path to the output audio file

        returns:
            None
        """
        with torch.inference_mode():
            waveform_out = self.audio_codec_model.decode(audio_codec)
        # print("waveform_out : ", waveform_out)
        # print("waveform_out.shape : ", waveform_out.shape)
        # print("waveform_out.cpu().squeeze(0) : ", waveform_out.cpu().squeeze(0))

        torchaudio.save(output_audio, waveform_out.cpu().squeeze(0), sample_rate=24000)
        print(f"[✓] Decoded audio saved: {output_audio}")


    def get_text_ids(self, text):
        """
        This function is used to get the text ids from the text.

        args:
            text: text to be encoded

        returns:
            text_ids: text ids
        """
        text = f"<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>"
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        return text_ids

    def get_speech_ids(self, speech_codes):


        # print(f"speech_codes: {type(speech_codes)}")
        # print(f"speech_codes: {speech_codes}")
        speech_ids = (
            [self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_START|>")]
            + [self.tokenizer.convert_tokens_to_ids(f"<|s_{code}|>") for code in speech_codes]
            + [self.tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")]
        )
        return speech_ids

    def get_text_and_speech_ids(self, text, audio_codes_list):
        speech_ids_status = True
        text_ids, speech_ids = self.get_text_ids(text), self.get_speech_ids(audio_codes_list)
        # print(f"text_ids: {text_ids}")
        # print(f"speech_ids: {speech_ids}")
        unk_id = self.tokenizer.unk_token_id

        if any(tid is None or tid == unk_id for tid in speech_ids):
            speech_ids_status = False
            raise ValueError("One or more speech_ids are None or unknown token IDs.")

        return text_ids, speech_ids, speech_ids_status



if __name__ == "__main__":
    # "duration": 5.920000076293945
    # Load SNAC model
    audio_codec_model = "hubertsiuzdak/snac_24khz"
    # "unsloth/Llama-3.2-1B-Instruct"
    tokenizer_name="Qwen/Qwen2.5-0.5B-Instruct"
    emotion_tag_path = "/media/sayan/hdd/TTS/QweenTTS/config/emotion_tag.json"
    ac = AudioCodec(
        code_path=audio_codec_model, 
        tokenizer_name=tokenizer_name, 
        codebook_size=4096,
        emotion_tag_path = emotion_tag_path,
        encode_status=True,
        decode_status=True,
        saving_pt_file=True,
        )
    input_dir = "/media/sayan/hdd/TTS/BanglaXTTS-dev-saiful/xtts_training/dataset/bn/HQ-STUDIO-TTS-7.16H/dataset"
    processed_folder_list = [
        # "codemixed_data",
        # "processed_sknahin"
        # "dataset_voice_5h",
        # "tts_studio_Hena",
        # "tts_studio_Maria",
        # "tts_studio_Rafa",
        # "tts_studio_Rakib",
        # "tts_audibles_100h"
        ]
    folder_list = os.listdir(input_dir)
    initial_time = time.time()
    for folder in folder_list:
        # if folder not in processed_folder_list:
        #     continue
        input_json_file = os.path.join(input_dir, folder, "label.json")
        print(f"[+] Processing: {input_json_file}")
        data = ac.load_json(input_json_file)
        start_time = time.time()
        log_duration = 0
        for d in tqdm(data):
            item = deepcopy(d)
            try:
                audio_path = item["audio_file"]
                text = item["text"]

                if "duration" in item:
                    duration = item["duration"]

                    log_duration+= duration
                    
                    if duration >  30:
                        continue

                audio_file_name = os.path.basename(audio_path)

                if folder == "codemixed_data":  
                    input_audio = os.path.join(input_dir, item["audio_file"])
                    output_json_file = os.path.join(input_dir, folder, "wavs", audio_file_name.replace(".wav", "_snac.json"))
                
                elif folder == "processed_sknahin":

                    input_audio = os.path.join(input_dir, item["audio_file"])
                    output_json_file = os.path.join(input_dir, folder, "json_snac", audio_file_name.replace(".wav", "_snac.json"))
                    os.makedirs(os.path.join(input_dir, folder, "json_snac"), exist_ok=True)
                
                else:
                    input_audio = os.path.join(input_dir, folder, "wavs", audio_file_name)
                    output_json_file = input_audio.replace(".wav", "_snac.json")

                
                if os.path.exists(output_json_file):
                    print(f"[+] Skipping: {output_json_file}")
                    continue

                if not os.path.exists(input_audio):
                    print(f"[+] Skipping: {input_audio}")
                    continue

                codec_codes = ac.snac_audio_encode(input_audio)
                audio_codes_list = ac.tokenise_audio(codec_codes)
                # decode_snac_format_codes = ac.get_decode_snac_format_codes(tokenised_audio)
                text_ids, speech_ids, speech_ids_status = ac.get_text_and_speech_ids(text, audio_codes_list)

                item["text_ids"] = text_ids
                item["speech_ids"] = speech_ids
                item["speech_ids_status"] = speech_ids_status
                new_item = {
                    "text": text,
                    "audio_file" : audio_path,
                    "text_ids" : text_ids,
                    "speech_ids" : speech_ids,
                    "speech_ids_status" : speech_ids_status
                }
                with open(output_json_file, 'w', encoding='utf-8') as json_file:
                    json.dump(new_item, json_file, indent=4, ensure_ascii=False)
                # print(f"[+] Saved: {output_json_file}")

            except Exception as e:
                print(f"Error: {e}")
                continue
        print(f"Total duration: {log_duration/3500} hours")
        print(f"Total time taken: {time.time() - start_time} seconds or {(time.time() - start_time)/3600} Hours")

    print(f"Processed {len(data)} files in {(time.time() - initial_time)/3600} Hours")

