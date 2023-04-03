import numpy
import pickle

class MlPipeline():
    def __init__(self, data) -> None:
        self.data = numpy.array(data).reshape(1, -1)
    
    def predict(self, choices):
        # Load the ML Models
        lr_model = pickle.load(open('.\ML Models\LogisticRegression_model.pkl', 'rb'))
        knn_model = pickle.load(open('.\ML Models\K-Nearest_Neighbours_model.pkl', 'rb'))
        svm_model = pickle.load(open('.\ML Models\SupportVectorMachine_model.pkl', 'rb'))

        # Load the scaler
        scaler = pickle.load(open('scaler.pkl', 'rb'))

        if choices[0] == 'Logistic Regression':
            return lr_model.predict(self.data)
        elif choices[0] == 'K-Nearest Neighbors':
            return knn_model.predict(self.data)
        else:
            return svm_model.predict(scaler.transform(self.data))