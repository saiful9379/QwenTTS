import os
import shutil
import glob
import json
import librosa
from tqdm import tqdm


def read_txt_file(file_path):

    with open(file_path, "r") as f:
        data = f.read().split("\n")
    data = [i for i in data if i.strip()]

    return data

def get_duration(file_path):
    return librosa.get_duration(path=file_path)


def is_numeric(value):
    try:
        float(value)  # Try converting to float
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    language = "bn"
    root_path = f"/media/sayan/hdd/TTS/BanglaXTTS-dev-saiful/xtts_training/dataset/bn/HQ-STUDIO-TTS-7.16H/dataset"
    # folders = ["Maria"]

    n_total_duraiton = 0
    for folder in os.listdir(root_path):
        # # # print(folder)
        # if folder not in folders:
        #     continue
        txt_file_path =  os.path.join(root_path, folder, "label.txt")
        output_json = os.path.join(root_path, folder, "label.json")

        data = read_txt_file(txt_file_path)
        data_list = []

        total_duraiton, discard_audio = 0, 0
        for line in tqdm(data):
            # if folder == "train":
            #     # print("line :", line.split("\t"))
            audio_name, text = line.split("|")[0],line.split("|")[1]

            audio_basename = audio_name
            if ".wav" in audio_basename:
                audio_root_path = os.path.join(root_path, folder, "wavs", audio_basename)
                n_audio_name = audio_basename
            else:
                audio_root_path = os.path.join(root_path, folder, "wavs", audio_basename+".wav")
                n_audio_name = audio_basename+".wav"

            if not os.path.exists(audio_root_path):
                # print("file not exit : ", audio_root_path)
                discard_audio+=1
                continue
            duration = get_duration(audio_root_path)
            # print(audio_name, text)
            chunk_data = {
                'text': text,
                'audio_file': f"{folder}/wavs/{n_audio_name}",
                'speaker_name':f"{folder}",
                'language':language,
                'duration': duration
            }
            data_list.append(chunk_data)
            total_duraiton += duration

            n_total_duraiton += duration

        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=4)

        print("total_duraiton : ", total_duraiton/3600)

        print("Total Number of file : ", len(data))
        print("discard file : ", discard_audio)

    print("total_duraiton : ", n_total_duraiton/3600)