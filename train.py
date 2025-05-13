"""
export CUDA_VISIBLE_DEVICES=6
torchrun --nproc_per_node=1 train.py configs/config_llama_3b.json
torchrun --nproc_per_node=1 --master-port=29501 train.py configs/config_llama_3b.json

torchrun --nproc_per_node=1 --master-port=29501 train.py config/config.json

"""


import os
import glob
import torch
import torch.nn as nn
import soundfile as sf
# from xcodec2.modeling_xcodec2 import XCodec2Model
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from dataclasses import dataclass, field
from typing import Optional, List
import sys
import transformers
import wandb
import json
from transformers.trainer_pt_utils import get_parameter_names
import numpy as np
import bitsandbytes as bnb
from liger_kernel.transformers import AutoLigerKernelForCausalLM
import random

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# SNAC codebook size
codebook_size = 4096

emotion_tag_path = "/media/sayan/hdd/TTS/QweenTTS/config/emotion_tag.json"

def _load_emotion_tag():
    emotion_tag = json.load(open(emotion_tag_path))
    emotion_tag_list = []
    for key, value in emotion_tag.items():
        emotion_tag_list.append(value)
    print(emotion_tag_list)
    return emotion_tag_list

emotion_tag_list = _load_emotion_tag()


@dataclass
class ModelArguments:
    # llm_model_name_or_path: Optional[str] = field(default="meta-llama/Llama-3.2-1B-Instruct")
    llm_model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-0.5B-Instruct")
    cache_dir: Optional[str] = field(default=None, metadata={"help": "Cache directory for the model."})

 
@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Root path to the memmap data."})

 
@dataclass
class CustomTrainingArguments(TrainingArguments):
    optim: str = field(default="adamw_torch_fused")
    model_max_length: int = field(
        default=1024,
        metadata={"help": "Maximum sequence length"},
    )
    logging_steps: int = field(default=100, metadata={"help": "Log every X updates"})
    report_to: Optional[str] = field(
        default=None, metadata={"help": "The integration to report the results and logs to."}
    )
    run_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the run for logging."}
    )
    gradient_checkpointing: bool = field(default=True)
    lr_scheduler_type: str = field(default="cosine", metadata={"help": "The learning rate scheduler to use."})
  
class TTSDataset(Dataset):
    def __init__(
        self,
        data_path,
        split,
        tokenizer,
    ):
        self.pad_token_id = tokenizer.pad_token_id
        self.tokenizer = tokenizer
        self.max_length = 1024
        self.ignore_index = -100
        print("Loading dataset...")

        print("data_path: ", data_path)
        print("split: ", split)
        # Use glob to search for files matching pattern for any rank and partial.
        # pattern = os.path.join(data_path, f"{split}_rank*_partial*_input_ids.memmap")
        pattern = os.path.join(data_path, f"{split}_input_ids.memmap")
        memmap_files = sorted(glob.glob(pattern))

        print(f"Found {len(memmap_files)} matching files for {split} split")
        print(f"memmap_files: {memmap_files}")

        self.chunks = []
        self.cum_lengths = [0]

        print("memmap_files: ", memmap_files)

        print("-------------------------------------")

        # If any matching files are found, load each one together with its shape.
        if memmap_files:
            for memmap_file in memmap_files:
                # Replace "input_ids.memmap" with "input_ids_shape.npy"
                shape_file = memmap_file.replace("input_ids.memmap", "input_ids_shape.npy")
                if not os.path.exists(shape_file):
                    raise ValueError(f"Shape file {shape_file} not found for {memmap_file}")
                shape = tuple(np.load(shape_file))
                chunk_memmap = np.memmap(
                    memmap_file, dtype="int32", mode="r", shape=shape
                )
                self.chunks.append(chunk_memmap)
                self.cum_lengths.append(self.cum_lengths[-1] + shape[0])
            self.length = self.cum_lengths[-1]
            print(f"Total length: {self.length}")
            print(f"cum_lengths: {self.cum_lengths}")
            print(f"chunks: {self.chunks}")
        else:
            # Fall back to the previous naming convention if no rank/partial files found.
            chunk_idx = 0
            self.chunks = []
            self.cum_lengths = [0]
            while True:
                # memmap_file = os.path.join(data_path, f'{split}_input_ids_{chunk_idx}.memmap')
                # shape_file = os.path.join(data_path, f'{split}_input_ids_{chunk_idx}_shape.npy')


                memmap_file = os.path.join(data_path, f'{split}_input_ids.memmap')
                shape_file = os.path.join(data_path, f'{split}_input_ids_shape.npy')



                if not os.path.exists(memmap_file) or not os.path.exists(shape_file):
                    break
                shape = tuple(np.load(shape_file))
                chunk_memmap = np.memmap(
                    memmap_file, dtype="int32", mode="r", shape=shape
                )
                self.chunks.append(chunk_memmap)
                self.cum_lengths.append(self.cum_lengths[-1] + shape[0])
                chunk_idx += 1

            if len(self.chunks) == 0:
                # Fallback to d single file strategy.
                memmap_file = os.path.join(data_path, f'{split}_input_ids.memmap')
                shape_file = os.path.join(data_path, f'{split}_input_ids_shape.npy')
                shape = tuple(np.load(shape_file))
                self.single_memmap = np.memmap(memmap_file, dtype='int32', mode='r', shape=shape)
                self.length = shape[0]
                self.chunks = [self.single_memmap]
                self.cum_lengths = [0, self.length]
            else:
                self.length = self.cum_lengths[-1]

        # Retrieve the special tokens.
        self.speech_generation_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_START|>')
        self.speech_generation_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_GENERATION_END|>')
        self.text_generation_start_id = tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_START|>')
        self.text_generation_end_id = tokenizer.convert_tokens_to_ids('<|TEXT_GENERATION_END|>')
        self.text_understanding_start_id = tokenizer.convert_tokens_to_ids('<|TEXT_UNDERSTANDING_START|>')
        self.text_understanding_end_id = tokenizer.convert_tokens_to_ids('<|TEXT_UNDERSTANDING_END|>')
        self.speech_understanding_start_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_START|>')
        self.speech_understanding_end_id = tokenizer.convert_tokens_to_ids('<|SPEECH_UNDERSTANDING_END|>')
        self.max_length = 1024
        self.ignore_index = -100  

    def __len__(self):
        return self.length

    def get_chunk_and_local_index(self, idx):
        # Find in which chunk the global index resides.
        for i in range(1, len(self.cum_lengths)):
            if idx < self.cum_lengths[i]:
                local_idx = idx - self.cum_lengths[i - 1]
                return self.chunks[i - 1], local_idx
        raise IndexError("Index out of bounds")

    def replace_tagged_token(self, token_list, target_token, new_sequence):
        idx = token_list.index(target_token)
        return token_list[:idx] + list(new_sequence) + token_list[idx+1:]

    def pad_sequence(self, sequence, max_length, value=0):
        if len(sequence) >= max_length:
            return sequence[:max_length]
        else:
            padding = torch.full((max_length - len(sequence),), value, dtype=sequence.dtype)
            return torch.cat([sequence, padding], dim=0)

    @classmethod
    def create_truncated_dataset(
        cls,
        data_path,
        split,
        tokenizer,
        num_samples=None,
        seed=None,
    ):
        dataset = cls(data_path, split, tokenizer)
        if num_samples is not None:
            # Set random seed for reproducibility.
            if seed is not None:
                np.random.seed(seed)
            total_length = len(dataset)
            indices = np.random.permutation(total_length)
            selected_indices = indices[:num_samples]
            dataset.selected_indices = sorted(selected_indices)
            dataset.use_selected = True
            dataset.length = len(dataset.selected_indices)
        else:
            dataset.use_selected = False

        return dataset

    def __getitem__(self, idx):
        # If using a truncated dataset, map idx to the original index.
        if hasattr(self, "use_selected") and self.use_selected:
            global_idx = self.selected_indices[idx]
        else:
            global_idx = idx

        chunk, local_idx = self.get_chunk_and_local_index(global_idx)
        input_ids_np = chunk[local_idx]
        input_ids = torch.tensor(input_ids_np, dtype=torch.long)
        labels = torch.full_like(input_ids, self.ignore_index)

 
        speech_gen_positions = (input_ids == self.speech_generation_start_id).nonzero(as_tuple=True)[0]
        text_gen_positions = (input_ids == self.text_generation_start_id).nonzero(as_tuple=True)[0]
 
        speech_gen_idx = speech_gen_positions[0].item()
        try:
            speech_gen_end_idx = (input_ids == self.speech_generation_end_id).nonzero(as_tuple=True)[0].item()
        except Exception as e:
            print(f"maybe Error in speech_gen_end_idx: {e}")
            speech_gen_end_idx = 1024
 
        text_sequence = input_ids[:speech_gen_idx]
        speech_sequence = input_ids[speech_gen_idx : speech_gen_end_idx + 1]
 
        # before we start, we need to choose a random speech_ids_prefix from the list
        
        chat = [
            {"role": "user", "content": "Convert the text to speech more human like without any leading silence:<|TEXT_UNDERSTANDING_START|>"},
            {"role": "assistant", "content": "<|SPEECH_GENERATION_START|>"}
        ]

        
        ids = self.tokenizer.apply_chat_template(chat, tokenize=True)

    
        ids = self.replace_tagged_token(ids, self.text_understanding_start_id, text_sequence)
        ids = self.replace_tagged_token(ids, self.speech_generation_start_id, speech_sequence)

        input_ids = torch.tensor(ids, dtype=torch.long)
        labels = torch.full_like(input_ids, self.ignore_index)

        try:
            speech_gen_idx_in_input = (
                input_ids == self.speech_generation_start_id
            ).nonzero(as_tuple=True)[0].item()
            labels[speech_gen_idx_in_input:] = input_ids[speech_gen_idx_in_input:]
        except Exception as e:
            print(f"maybe Error in speech_gen_idx_in_input: {e}")
            # speech_gen_idx_in_input = len(input_ids) - 1
 
            labels  = input_ids 

 
        attention_mask = (input_ids != self.pad_token_id).long()
 
        labels[input_ids == self.pad_token_id] = self.ignore_index

 
        
        input_ids = self.pad_sequence(input_ids, self.max_length, value=self.pad_token_id)
        attention_mask = self.pad_sequence(attention_mask, self.max_length, value=0)
        labels = self.pad_sequence(labels, self.max_length, value=self.ignore_index)

        return {
            'input_ids': list(input_ids),
            'labels': list(labels),
            'attention_mask': list(attention_mask)
        }
 

def main():
    # Parse arguments
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, CustomTrainingArguments))
    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        # Load arguments from the specified JSON file
        (
            model_args,
            data_args,
            training_args,
        ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # Attempt to load arguments from the default 'config.json' file
        default_config_file = 'config.json'
        if os.path.exists(default_config_file):
            (
                model_args,
                data_args,
                training_args,
            ) = parser.parse_json_file(json_file=os.path.abspath(default_config_file))
        else:
            # If 'config.json' does not exist, parse arguments from the command line
            (
                model_args,
                data_args,
                training_args,
            ) = parser.parse_args_into_dataclasses()

    is_main_process = training_args.local_rank in [-1, 0]
 
    if training_args.report_to == "wandb" and is_main_process:
        wandb.init(
            project="llm_tts",  
            config=training_args.to_sanitized_dict(),
            name=training_args.run_name
        )

 
 
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        # Find all checkpoint directories in the output directory
        checkpoints = [os.path.join(training_args.output_dir, d) for d in os.listdir(training_args.output_dir) if d.startswith("checkpoint-")]
        if len(checkpoints) > 0:
            # Get the most recent checkpoint based on modification time
            last_checkpoint = max(checkpoints, key=os.path.getmtime)

    if last_checkpoint is not None:
        print(f"Loading model and tokenizer from checkpoint {last_checkpoint}")
        tokenizer = AutoTokenizer.from_pretrained(last_checkpoint, token=os.getenv("HF_TOKEN"))
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
        model = AutoLigerKernelForCausalLM.from_pretrained(last_checkpoint, 
                                                           token=os.getenv("HF_TOKEN"),
                                                        #    attn_implementation="flash_attention_2",
                                                           torch_dtype='bfloat16'
                                                           )
    else:
        print("No checkpoint found, starting training from scratch")
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.llm_model_name_or_path,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
        
        original_vocab_size = len(tokenizer)
        print(f"Original tokenizer vocabulary size: {len(tokenizer)}")

        Start_End_tokens = [
            '<|TEXT_GENERATION_START|>',
            '<|TEXT_GENERATION_END|>',
            '<|TEXT_UNDERSTANDING_START|>',
            '<|TEXT_UNDERSTANDING_END|>',
            '<|SPEECH_GENERATION_START|>',
            '<|SPEECH_GENERATION_END|>',
            '<|SPEECH_UNDERSTANDING_START|>',
            '<|SPEECH_UNDERSTANDING_END|>'
        ]

        speech_token_lenght = 7 * codebook_size

        new_speech_tokens = [f'<|s_{i}|>' for i in range(speech_token_lenght)]   
        all_new_tokens = Start_End_tokens + emotion_tag_list + new_speech_tokens
        num_added_tokens = tokenizer.add_tokens(all_new_tokens)
        print(f"Added {num_added_tokens} speech tokens to the tokenizer.")

        # Use the tokenizer's existing special tokens instead of hardcoding IDs
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        tokenizer.save_pretrained(training_args.output_dir)
 
        model = AutoLigerKernelForCausalLM.from_pretrained(model_args.llm_model_name_or_path, 
                                                           cache_dir=model_args.cache_dir,
                                                        #    attn_implementation="flash_attention_2",
                                                           torch_dtype='bfloat16'
                                                           )

        # Adjust the embedding layer and lm_head
        model.resize_token_embeddings(len(tokenizer))
        model.config.vocab_size = len(tokenizer)  # Update config vocab size too

        # Verify the size of the embedding layer
        print(f"Tokenizer vocabulary size: {len(tokenizer)}")
        print(f"Model's embedding layer size: {model.model.embed_tokens.weight.size(0)}")
        print(f"Model's lm_head size: {model.lm_head.weight.size(0)}")

        # Make sure to set padding_side to 'left' for the TTSDataset as well
        tokenizer.padding_side = 'right'

 
    train_dataset = TTSDataset(
        data_path=data_args.data_path,
        split="train",
        tokenizer=tokenizer,
    )
    print(f"Train dataset length: {len(train_dataset)}")
    _ = train_dataset[0]  # quick check
    print(train_dataset)

    # exit()

    # Automatically load evaluation files; if none found, set to None.
    eval_dataset = None

    split = "test"
    # eval_pattern = os.path.join(data_args.data_path, f"test_rank*_partial*_input_ids.memmap")

    eval_pattern = glob.glob(data_args.data_path+f"/{split}_input_ids.memmap")
    if eval_pattern:
        eval_dataset = TTSDataset.create_truncated_dataset(
            data_path=data_args.data_path,
            split="test",
            tokenizer=tokenizer,
            num_samples=12,
            seed=42,
        )

    data_collator = default_data_collator

    ##########################################
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    optimizer_kwargs = {
        "betas": (training_args.adam_beta1, training_args.adam_beta2),
        "eps": training_args.adam_epsilon,
    }

    optimizer_kwargs["lr"] = training_args.learning_rate

    adam_bnb_optim = bnb.optim.Adam8bit(
        optimizer_grouped_parameters,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        lr=training_args.learning_rate,
    )
    training_args.optim = (adam_bnb_optim, None)
    training_args.optimizers = (adam_bnb_optim, None)




    training_args.bf16 = True
    training_args.use_bfloat16 = True
    training_args.non_blocking = True
    ##### Optimizer #####

    print("Training Parameters:")
    print(training_args)
    print("===========")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        optimizers=(adam_bnb_optim, None),
    )
    if is_main_process:
        trainer.add_callback(transformers.integrations.WandbCallback())

 
    trainer.train(resume_from_checkpoint=last_checkpoint)
 
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()
