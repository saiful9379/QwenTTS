import os
import json
import glob
from tqdm import tqdm

def load_json_file(json_file_path):
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_json_file(json_data, json_file_path):
    with open(json_file_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    input_data = "/media/sayan/hdd/TTS/BanglaXTTS-dev-saiful/xtts_training/dataset/bn/HQ-STUDIO-TTS-7.16H/dataset"


    folder_list = os.listdir(input_data)


    text_lenght,  speech_id_lenght = 0, 0 

    for folder in folder_list:
        print(folder)
        all_data = []
        discard = 0
        json_files = glob.glob(os.path.join(input_data, folder, "wavs", "*_snac.json"), recursive=True)
        for json_file in tqdm(json_files):
            try:
                data = load_json_file(json_file)

                # print("data : ", data)
                text_id = data["text_ids"]
                speech_id = data["speech_ids"]

                if len(text_id) > text_lenght:
                    text_lenght = len(text_id)
                if len(speech_id) > speech_id_lenght:
                    speech_id_lenght = len(speech_id)
            
                all_data.append(data)
            except Exception as e:
                print(e)
                print(json_file)
                discard += 1
                continue
        print(f"discard : {discard}")
        print(f"total : {len(all_data)}")
        with open(os.path.join(input_data, folder, "label_qween_snac.json"), "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=4)

    print("text_lenght : ", text_lenght)
    print("speech_id_lenght : ", speech_id_lenght)