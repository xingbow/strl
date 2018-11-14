import csv
import datetime
import matplotlib.pyplot as plt
import OIModel as om
import time
import xgboost as xgb
import numpy as np

startTime = 1388505600
time1 = time.time()
# type in file loc
transitionDataLoc = 'C:/Users/Tony Xu/OneDrive/KDD 2/test.csv' # transition file
weatherDataLoc = 'C:/Users/Tony Xu/OneDrive/KDD 2/weather.csv' # weather info file

# Initialize function, no need to modify
hisInputData, transitionMatrixDuration, transitionMatrixDetination, weatherMatrix ,entireSituation,weatherMatrixAll\
    = om.read_inData(transitionDataLoc,weatherDataLoc)
timeStampBound = 1361667600
overAllTest, pred_y, dayNumBound = om.predicted_overallSituation(dayNumBound, entireSituation, weatherMatrixAll)

def test():
    # Inputs time stamp : e.g. 1373964540
    # Inputs station ID : e.g.212
    # return: expected departure(rent) number of this station in a period(hour)
    timeStamp = 1393465860
    startStationID = 11
    dayNumBound = (timeStampBound-startTime)//(24*3600)
    expectedDepartureTime = om.get_expectDepartureNumber(timeStamp,startStationID,hisInputData,weatherMatrix,overAllTest,pred_y,dayNumBound)

    # Inputs time stamp : e.g. 1373964540
    # Inputs station ID : e.g.212
    # return: expected destination and departure time
    # return: reference result is the same as predictedDestination except that it uses string time, thus more intuitive



    print('Expected departure time for',startStationID,'is:')
    for var in expectedDepartureTime:
        print(var)
    print('length:',len(expectedDepartureTime))
    print('..............')
