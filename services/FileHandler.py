import os
import pandas as pd
from fastapi import HTTPException


class FileHandler:
    def __init__(self, path: str):
        self.path = path

    def upload(self, file_path: str):
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")

        self.path = file_path

        try:
            data = pd.read_excel(self.path)
            return data.values
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))