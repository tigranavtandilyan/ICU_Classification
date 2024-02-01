import argparse
import pickle
import pandas as pd
from preprocessor import Preprocessor
from model import Model

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true")
parser.add_argument("--data_path", action="store")
args = parser.parse_args()
preprocessor = Preprocessor()
model = Model()
if args.test:
    data = pd.read_csv(args.data_path)
    loaded_preprocessor = pickle.load(open('finalized_preprocessor.sav', 'rb'))
    new_data = loaded_preprocessor.transform(data)
    loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
    loaded_model.save_probas(new_data)
else:
    data = pd.read_csv(args.data_path)
    target = data["In-hospital_death"]
    features = data.drop(columns=["In-hospital_death"])
    preprocessor.fit(features)
    filename_prep = 'finalized_preprocessor.sav'
    pickle.dump(preprocessor, open(filename_prep, 'wb'))
    features = preprocessor.transform(features)
    model.fit(features, target)
    filename_model = 'finalized_model.sav'
    pickle.dump(model, open(filename_model, 'wb'))
