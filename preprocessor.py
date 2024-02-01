from sklearn.preprocessing import StandardScaler


class Preprocessor:
    def __init__(self):
        self.data = None
        self.norms = {"Gender": 0.5, "Albumin": 4.3, "ALP": 72, "ALT": 29, "AST": 23, "Bilirubin": 0.3,
                      "BUN": 14, "Cholesterol": 190, "Creatinine": 1.0, "FiO2": 0.2, "GCS": 15, "Glucose": 100,
                      "HCO3": 24, "HCT": 45, "HR": 72, "K": 3.9, "Lactate": 0, "Mg": 2, "MAP": 90,
                      "MechVent": 0, "Na": 140, "PaO2": 100, "PaCO2": 40, "pH": 7.4, "Platelets": 300, "RespRate": 15,
                      "SaO2": 100, "Temp": 36.6, "Troponin": 0, "WBC": 8, "Dias": 80, "Sys": 120, "Urine": 25,
                      "Height": 170, "Weight": 70}

        self.scaler = StandardScaler()

    def fit(self, data):
        self.data = data
        self.data = self.pretransform(data)
        self.scaler.fit(self.data)

    def pretransform(self, data):
        if data.isna().mean().sum() > 0:
            data = data.drop(columns=["recordid"])
            for norm in self.norms:
                for col in data.columns:
                    if norm in col:
                        data[col] = data[col].fillna(self.norms[norm])
        return data

    def transform(self, data):
        data = self.pretransform(data)
        return self.scaler.transform(data)
