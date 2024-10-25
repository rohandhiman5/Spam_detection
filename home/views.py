from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from django.conf.urls.static import static

# Create your views here.
import pickle


# Load model and vectorizer once, globally
model = pickle.load(open("static/ML/NB.pkl", "rb"))
vectorizer = pickle.load(open("static/ML/Vectorizer_logistic_regression.pkl", "rb"))  # Added encoding


def index(request):
    return render(request,'index.html')


def getPrediction(mail):
    
    input_vector=vectorizer.transform([mail])

    prediction=model.predict(input_vector)

    return prediction[0]




def result(request):
    input=request.POST.get('message')

    prediction=getPrediction(input)

    return render(request,'result.html',{'output':prediction})








