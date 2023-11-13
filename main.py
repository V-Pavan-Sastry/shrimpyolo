from fastapi import FastAPI,UploadFile,HTTPException
from PIL import Image
from io import BytesIO
import numpy as np
import torch
app=FastAPI()
@app.get("/")
def read():
    return {"Shrimp Detector": "Activated"}

@app.post("/detect")
def runyolo(img: UploadFile):
    if img.filename.split(".")[-1]in("jpg","jpeg","png"):
        pass
    else:
        raise HTTPException(status_code=415,detail="item not found")
    img=Image.open(BytesIO(img.file.read()))
    img=np.array(img)
    model=torch.hub.load('ultralytics/yolov5','custom',path='./best.pt')
    results=model(img)
    print(results.xyxyn[0])
    detection={}
    res=results.xyxyn[0].tolist()
    print(res)
    for i in range(len(res)):
        detect={}
        if int(res[i][-1])==1:
            detect["name"]="shrimp"
            detect["confidence"]=str(int(res[i][0]*100))+"%"
            detect["xmin"]=str(res[i][0])
            detect["ymin"]=str(res[i][0])
            detect["xmax"]=str(res[i][0])
            detect["ymax"]=str(res[i][0])
            detection[str(i)]  = detect
        else:
            detect["name"]="fish"
            detect["confidence"]=str(int(res[i][0]*100))+"%"
            detect["xmin"]=str(res[i][0])
            detect["ymin"]=str(res[i][0])
            detect["xmax"]=str(res[i][0])
            detect["ymax"]=str(res[i][0])
            detection[str(i)]  = detect
    print(detection)
    return detection