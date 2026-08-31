#Process Video to mp3 format using FFMPEG
import os
import subprocess

files = os.listdir("videos")

for file in files:
    # Skip hidden files like .DS_Store and files that don't match the pattern
    # Can be changed later based on the use case
    # for now i have only some files to process so i dont need to check for the
    #specific charcater that is not in the file ex : [ and #
    # so i am removing code : or " [" not in file or " #" not in file
    # from below
    if file.startswith("."):
        continue
    try:
        # Extract the tutorial number
        tutorial_num = file.split(" [")[0].split(" #")[1]
        file_name = file.split(" ｜ ")[0]
        print(f"Extracted Number: {tutorial_num}")
        print(tutorial_num,file_name)
        subprocess.run(["ffmpeg","-i",f"videos/{file}",f"audios/{tutorial_num}_{file_name}.mp3"])


    except IndexError:
        print(f"Failed to parse format for: {file}")
