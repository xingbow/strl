import csv
import datetime
import matplotlib.pyplot as plt
import OIModel as om
import time
import xgboost as xgb
import numpy as np

startTime = 1375286400
time1 = time.time()
# type in file loc
transitionDataLoc = 'C:/Users/Tony Xu/OneDrive/KDD 2/Final-2013-08 - Citi Bike trip data.csv2013-09 - Citi Bike trip data.csv' # transition file
weatherDataLoc = 'C:/Users/Tony Xu/OneDrive/KDD 2/weather.csv' # weather info file

# Initialize function, no need to modify
hisInputData, transitionMatrixDuration, transitionMatrixDetination, weatherMatrix ,entireSituation,weatherMatrixAll\
    = om.read_inData(transitionDataLoc,weatherDataLoc)

timeStamp = 1380243600
timeStampBound = timeStamp
dayNumBound = (timeStampBound - startTime) // (24 * 3600)

overAllTest, pred_y, dayNumBound = om.predicted_overallSituation(dayNumBound, entireSituation, weatherMatrixAll)

def test():
    # Inputs time stamp : e.g. 1373964540
    # Inputs station ID : e.g.212
    # return: expected departure(rent) number of this station in a period(hour)

    startStationID = 11
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
'''
def draw_real(allPredict,timeStampStart,dayDrawNum,stationID,allPredictNum):
    f = open(transitionDataLoc)
    reader = csv.reader(f)  # 444,285,236
    dayStart = int((timeStampStart -startTime )//(3600*24))

    realRent = np.zeros(dayDrawNum * 16)
    realRent2 = np.zeros(dayDrawNum * 16)

    for row in reader:
        if row[0] != 'tripduration':
            timeStamp = int(row[7]) + 1372608000
            dayNow = int((timeStamp - startTime) // (3600 * 24))
            #print(dayNow,dayStart,dayDrawNum)
            if (dayNow>=dayStart)&(dayNow<dayStart+dayDrawNum)&(int(row[10])==stationID):
                periodNum = int((timeStamp - startTime) // 3600 % 24) - 7
                realRent[16*(dayNow-dayStart)+periodNum] += 1

    for timeStampP in allPredict:
        dayNow = int((timeStampP - startTime) // (3600 * 24))
        periodNum = int((timeStampP - startTime) // 3600 % 24) - 7
        realRent2[16 * (dayNow - dayStart) + periodNum] += 1

    plt.figure(2)
    plt.plot(realRent)
    plt.plot(realRent2)
    plt.show()

    print('Accuracy:',sum(abs(realRent-realRent2))/sum(realRent)*100,'%')

def draw_All(stationID,timeStampStart):
    f = open(transitionDataLoc)
    reader = csv.reader(f)  # 444,285,236
    dayStart = int((timeStampStart - startTime) // (3600 * 24))

    realRent = np.zeros(60 * 16)

    for row in reader:
        if row[0] != 'tripduration':
            timeStamp = int(row[14]) + 1372608000
            dayNow = int((timeStamp - startTime) // (3600 * 24))
            if int(row[17]) == stationID:
                periodNum = int((timeStamp - startTime) // 3600 % 24) - 7
                realRent[16 * (dayNow - dayStart) + periodNum] += 1
    plt.figure(1)
    plt.plot(realRent)
    plt.show()


def test2(overAllTest,pred_y,timeBound):
    timeStampStart = 1380243600

    startStationID = 9

    #draw_All(startStationID,timeStampStart)

    allPredict = []
    allPredictNum = []
    dayDrawNum = 4
    periodStart = int((timeStampStart - startTime) // 3600 % 24)
    for i in range(dayDrawNum):
        for j in range(16):
            #print(timeStampStart - timeStampStart % (24*3600))
            timeStamp = (timeStampStart - (timeStampStart-startTime)%(24*3600)) + (j+7) * 3600 + i * 24 * 3600
            expectedDepartureTime = om.get_expectDepartureNumber(timeStamp,startStationID,hisInputData,weatherMatrix,overAllTest,pred_y,timeBound)
            allPredict = allPredict + expectedDepartureTime
            allPredictNum.append(len(expectedDepartureTime))
    draw_real(allPredict,timeStampStart,dayDrawNum,startStationID,allPredictNum)



overAllTest,pred_y,dayNumBound = om.predicted_overallSituation(55,entireSituation,weatherMatrixAll)
test2(overAllTest,pred_y,dayNumBound)
'''
test()
time2 = time.time()
print('Time used:',time2-time1,'s')
