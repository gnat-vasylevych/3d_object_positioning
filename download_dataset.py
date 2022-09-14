import requests
import os

if __name__ == "__main__":
    public_url = "https://storage.googleapis.com/objectron"
    class_to_download = "cup_annotations"  # there are several classes available. for more info check google objectron
    blob_path = public_url + "/v1/index/" + class_to_download
    video_ids = requests.get(blob_path).text
    video_ids = video_ids.split('\n')  # list of available videos

    download_directory = "cup_annotations"  # directory where to download videos

    parent_directory = os.getcwd()

    path = os.path.join(parent_directory, download_directory)

    try:
        os.mkdir(path)
    except FileExistsError:
        print("Directory 'cup_annotations' already exists")

    os.chdir(path)

    # Download all videos in cup test dataset
    for i in range(1):
        video_filename = public_url + "/videos/" + video_ids[i] + "/video.MOV"
        annotation_filename = public_url + "/annotations/" + video_ids[i] + ".pbdata"
        video = requests.get(video_filename)
        annotation = requests.get(annotation_filename)

        file = open("{}/video.MOV".format(video_ids[i]).replace('/', '_'), "wb")
        file.write(video.content)
        file.close()

        with open("{}/annotation.pbdata".format(video_ids[i]).replace('/', '_'), 'wb') as file:
            file.write(annotation.content)

    os.chdir(parent_directory)
