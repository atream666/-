from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pickle
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from fastapi.responses import FileResponse
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

df = pd.DataFrame({
    'area': [80, 120, 150, 200],
    'rooms': [2, 3, 4, 5],      
    'age': [5, 10, 20, 25],
    'price': [100, 180, 220, 350]
})

model = None

@app.get('/data')
def get_data():
    return {
        'columns': list(df.columns),
        'rows': df.shape[0],
        'preview': df.head(20).to_dict(orient='records'),
    }

@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    global df
    df = pd.read_csv(file.file)
    return {
        'status': 'ok',
        'rows': len(df)
    }

@app.post('/train')
def train_model():
    global df, model
    
    required_cols = {'area', 'rooms', 'age', 'price'}
    
    if not required_cols.issubset(set(df.columns)):
        return {'error': f"数据缺少必需列: {required_cols - set(df.columns)}"}

    X = df[['area', 'rooms', 'age']] 
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    pickle.dump(model, open('model.pkl', 'wb'))
    return {
        'mae': mae,
        'rmse': rmse,
        'message':'模型训练成功'
    }

class PredictInput(BaseModel):
    area: float
    rooms: int
    age: float

@app.post('/predict')
def predict_value(input: PredictInput):
    try:
        model = pickle.load(open('model.pkl', 'rb'))
    except:
        return {'error':' 模型未训练，点击/train'}
        
    X_new = [[input.area, input.rooms, input.age]]
    
    prediction = model.predict(X_new)[0]

    return {'prediction': float(prediction)}

@app.get('/generate-report')
def generate_report(area: float,
                    rooms: int,
                    age: float,
                    prediction: float):
    try:
        pdfmetrics.registerFont(TTFont("SimHei", "C:/Windows/Fonts/simhei.ttf"))
        font_name = "SimHei"
    except:
        font_name = "Helvetica"
        
    filename = 'report.pdf'
    c = canvas.Canvas(filename)
    c.setFont(font_name, 20)
    c.drawCentredString(300, 800, "机器学习预测报告")
    
    c.setFont(font_name, 12)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.drawString(50, 770, f"报告生成时间：{timestamp}")

    c.drawString(50, 740, "输入参数：")
    c.drawString(70, 720, f"面积（area）：{area} 平方米")
    c.drawString(70, 700, f"房间数（rooms）：{rooms} 间")
    c.drawString(70, 680, f"屋龄（age）：{age} 年")

    c.setFont(font_name, 14)
    c.drawString(50, 650, f"预测房价：{prediction:.2f} 万元")

    c.showPage()
    c.save()

    return FileResponse(
        filename,
        media_type="application/pdf",
        filename="ML_Report.pdf"
    )