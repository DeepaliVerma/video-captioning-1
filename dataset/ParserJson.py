import json

badVideos=[250,5691,1054,6575,5596,1648]
goodVideos=[i for i in range(0,7010)]#[0,6513,7010]
chooseVideos=[]
for video in goodVideos:
    if video not in badVideos:
        chooseVideos.append("video{}".format(video))

if __name__=="__main__":
    jsonPath="/mnt/MSR-VTT/train_val_videodatainfo.json"
    newJson={}
    for video in chooseVideos:
        newJson[video]={}
    with open(jsonPath, encoding='utf-8-sig', errors='ignore') as file:
        jsonContent = json.load(file, strict=False)
        info=jsonContent["info"]
        videos=jsonContent["videos"]
        sentences=jsonContent["sentences"]
        for video in videos:
            if video["video_id"]  in chooseVideos:
                newJson[video["video_id"]]["category"] = video["category"]
                newJson[video["video_id"]]["split"] = video["split"]
                newJson[video["video_id"]]["time"] = int(video["end time"] - video["start time"])
                newJson[video["video_id"]]["caption"] = []
        for sentence in sentences:
            if sentence["video_id"] in chooseVideos:
                newJson[sentence["video_id"]]["caption"].append(sentence["caption"])

    saveJson = json.dumps(newJson)
    jsonFile = open("/mnt/MSR-VTT/MSR-VTT-total.json", 'w')
    jsonFile.write(saveJson)
    jsonFile.close()




    # jsonPath = "/mnt/MSR-VTT/MSR-VTT-processedjson.json"
    # with open(jsonPath, encoding='utf-8-sig', errors='ignore') as file:
    #     jsonContent = json.load(file, strict=False)




