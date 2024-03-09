from fastapi import FastAPI, UploadFile, File
import inference

app = FastAPI()

@app.get("/")
def root():
    return {"response": "You are at root."}

@app.post("/api/get_predictions")
async def get_predictions(file: UploadFile = File(...)):

    print("########### READING IMAGE FILE  ##################", file.filename, "\n")
    
    content = await file.read()

    print("########### PREDICTION IN PROGRESS  ##################", file.filename, "\n")
    
    predicted_label = await inference.get_prediction(content)

    print("########### PREDICTION SUCCESSFUL  ##################")
        
    return predicted_label