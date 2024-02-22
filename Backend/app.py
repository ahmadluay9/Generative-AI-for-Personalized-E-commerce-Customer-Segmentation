import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

from model.model import model
from ai import chain, ai_response

app= FastAPI()

class User(BaseModel):
    count_orders: int
    average_spend: float
    return_ratio : float

class Info(BaseModel):
    cust_info: str

@app.get('/')
def index():
    return {'message': 'Customer Churn Prediction Application'}

@app.post('/predict')
def prediction(data: User):
    try:
        # Convert input data to pandas DataFrame
        df = pd.DataFrame([data.dict()])

        # Make prediction using the loaded model
        prediction_result = model.predict(df)

        # Print prediction for debugging
        print("Prediction:", prediction_result)

        # Extract the prediction result
        result = int(prediction_result[0])

        return result
    except Exception as e:
        print(e)  # Print the exception traceback
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/ai")
async def get_answer(cust_info: Info):
    response = chain.run(
    cust_info = cust_info.cust_info,
    ai_response= ai_response
    )

    return {response}
 

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)