from fastapi import FastAPI, HTTPException
from meteostat import Point, Daily
from datetime import datetime, timedelta
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Weather API")

class Location(BaseModel):
    lat: float
    lon: float
    days: int = 7

@app.get("/")
def read_root():
    return {"message": "Weather API jest aktywne"}

@app.post("/weather/")
def get_weather(location: Location):
    try:
        # Ustawienie punktu
        point = Point(location.lat, location.lon)
        
        # Ustawienie zakresu czasu
        end = datetime.now()
        start = end - timedelta(days=location.days)
        
        # Pobranie danych
        data = Daily(point, start, end)
        data = data.fetch()
        
        # Konwersja do formatu JSON
        result = []
        for date, row in data.iterrows():
            result.append({
                "date": date.strftime("%Y-%m-%d"),
                "tavg": row.get('tavg'),
                "tmin": row.get('tmin'),
                "tmax": row.get('tmax'),
                "prcp": row.get('prcp'),
                "snow": row.get('snow'),
                "wspd": row.get('wspd')
            })
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)