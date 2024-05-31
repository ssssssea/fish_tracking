import os
import subprocess

#conda_act_command = "conda activate yolotracking"

track_command = "python3 track.py --yolo-model weights/yolo_nas_bestv0002.pth --tracking-method bytetrack --save --save-txt --exist-ok --source "
name_command = "--name"
# 각 명령어 실행
parent_folder_path = '/mnt/storage1/dataset/smart_fish_farm/fish_videos/crowd_video_histogram_round3/'

#subprocess.run(conda_act_command, shell=True)
states = ["hungry", "stuffed"]

for i in range(1, 10):
    # 현재 숫자 폴더 경로
    folder_path_num = os.path.join(parent_folder_path, str(i))

    for state in states:
        folder_path = os.path.join(folder_path_num, state)
        print(folder_path)
        files = os.listdir(folder_path)

    # 각 파일에 대해 실행
        for file in files:
        # 파일 확장자가 비디오 형식인 경우에만 실행 (예: mp4, avi 등)
            if file.endswith(('.mp4', '.avi', '.mov')):
                file_path = os.path.join(folder_path, file)
                command = f"{track_command} {file_path} {name_command} crowd_{i}_{state}"
                subprocess.run(command, shell=True)

