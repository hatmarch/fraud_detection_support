import joblib

class FraudPredict(object):

    def __init__(self):
        self.model = joblib.load('model.pkl')
        self.class_names = ['V3','V4','V10','V11','V12','V14','V17'];

    def predict(self,X,features_names):
        return self.model.predict_proba(X)
