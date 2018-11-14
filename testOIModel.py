import csv
import datetime
import matplotlib.pyplot as plt
import OIModel as om
import time

startTime = 1391184000
time1 = time.time()
# type in file loc
transitionDataLoc = 'C:/Users/Tony Xu/OneDrive/KDD 2/test.csv' # transition file
weatherDataLoc = 'C:/Users/Tony Xu/OneDrive/KDD 2/weather.csv' # weather info file

# Initialize function, no need to modify
hisInputData, transitionMatrixDuration, transitionMatrixDetination, weatherMatrix = om.read_inData(transitionDataLoc,weatherDataLoc)

def test():
    # Inputs time stamp : e.g. 1373964540
    # Inputs station ID : e.g.212
    # return: expected departure(rent) number of this station in a period(hour)
    timeStamp = 20634634+1372608000
    startStationID = 22
    expectedDepartureTime = om.get_expectDepartureNumber(timeStamp,startStationID,hisInputData,weatherMatrix)

    # Inputs time stamp : e.g. 1373964540
    # Inputs station ID : e.g.212
    # return: expected destination and departure time
    # return: reference result is the same as predictedDestination except that it uses string time, thus more intuitive
    timeStamp = 20634634 + 1372608000
    startStationID = 22
    predictedDestination = om.get_predictedDestination(timeStamp,startStationID,transitionMatrixDuration, transitionMatrixDetination)

    # Inputs time stamp : e.g. 1373964540
    # Inputs start station ID : e.g.212
    # Inputs end station ID : e.g.404
    # return: expected time used
    timeStamp = 20634634 + 1372608000
    startStationID = 22
    endStationID = predictedDestination
    predictedDuration = om.get_predictedDuration(timeStamp,startStationID,endStationID ,transitionMatrixDuration, transitionMatrixDetination)


    print('Expected departure time for',startStationID,'is:')
    for var in expectedDepartureTime:
        print(var)
    print('length:',len(expectedDepartureTime))
    print('..............')
    print('Expected destination for',startStationID,':',predictedDestination)
    print('..............')
    print('Expected duration for',startStationID,'to',endStationID,':',predictedDuration,'s')
    print('..............')

def draw_real(allPredict,timeStampStart,dayDrawNum,stationID):
    f = open(transitionDataLoc)
    reader = csv.reader(f)  # 444,285,236
    dayStart = int((timeStampStart -startTime )//(3600*24))
    counter = 0
    hisRecord = []
    timeRecord = []

    counter2 = 0
    hisRecord2 = []
    timeRecord2 = []
    for row in reader:
        if row[0] != 'tripduration':
            timeStamp = int(row[14]) + 1372608000
            dayNow = int((timeStamp -startTime ) // (3600 * 24))
            if (dayNow>=dayStart)&(dayNow<dayStart+dayDrawNum)&(int(row[17])==stationID):
                counter -= 1
                hisRecord.append(counter)
                timeRecord.append(int(row[14]))
            elif(dayNow>=dayStart)&(dayNow<dayStart+dayDrawNum)&(int(row[18])==stationID):
                counter += 1
                hisRecord.append(counter)
                timeRecord.append(int(row[14]))

    timeStamp = int(row[14]) + 1372608000
    dayNow = int((timeStamp - startTime) // (3600 * 24))
    if (dayNow >= dayStart) & (dayNow < dayStart + dayDrawNum) & (int(row[17]) == stationID):
        counter -= 1
        hisRecord.append(counter)
        timeRecord.append(int(row[14]))
    elif (dayNow >= dayStart) & (dayNow < dayStart + dayDrawNum) & (int(row[18]) == stationID):
        counter += 1
        hisRecord.append(counter)
        timeRecord.append(int(row[14]))

    plt.plot(timeRecord,hisRecord)
    plt.show()

test()

time2 = time.time()
print('Time used:',time2-time1,'s')
