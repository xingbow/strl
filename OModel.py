import numpy as np
import csv
import datetime
import time
import math
import random
import matplotlib.pyplot as plt

'''
Define 5 episode
episode1 = [7,11]
episode2 = [12,16]
episode3 = [17,22]
episode4 = [[11,12],[16,17]]
episode5 = [22,23]'''

'''Loading files'''
weatherDataLoc = 'C:/Users/Tony Xu/OneDrive/KDD 2/test0.csv'
transitionDataLoc = 'C:/Users/Tony Xu/OneDrive/KDD 2/test222.csv'
stationStatusDataLoc = 'C:/Users/Tony Xu/OneDrive/KDD 2/stationStatus.csv'

startTime = 1372608000  # 2013/7/1 0:00
matrixSize = 3003

hisInputDataGlobal = []
transitionMatrixDurationGlobal = []
transitionMatrixDetinationGlobal = []
weatherMatrixGlobal = []
stationStatusMatrixGlobal = []


def get_timeString(timeStamp):
    nowTime = datetime.datetime.fromtimestamp(timeStamp)
    timeString = nowTime.strftime("%Y/%m/%d %H:%M:%S")
    return timeString



def read_inData():
    '''

    '''

    '''Initialize'''
    [loadDuration, loadStartTime, loadEndTime, loadStartStaionID, loadEndStationID, loadClusterTag] = [0, 1, 2, 3, 5, 6]

    '''!!!Do not modify this part!!!'''
    def get_attributes(timeStamp):
        '''
        return True if yes, False if no
        return # day caompared with start time
        return episode number
        return period number compared with start time
        '''
        dayNum = ((timeStamp - startTime) // (3600 * 24))
        if (dayNum % 7 == 5) | (dayNum % 7 == 6):
            weekdayFlag = False
        else:
            weekdayFlag = True
        hourEpisodeNum = (timeStamp - startTime) % (3600 * 24) // 3600
        if (hourEpisodeNum >= 7) & (hourEpisodeNum < 11):
            episodeNum = 1
        elif (hourEpisodeNum >= 12) & (hourEpisodeNum < 16):
            episodeNum = 2
        elif (hourEpisodeNum >= 17) & (hourEpisodeNum < 22):
            episodeNum = 3
        elif ((hourEpisodeNum >= 11) & (hourEpisodeNum < 12)) | (hourEpisodeNum >= 16) & (hourEpisodeNum < 17):
            episodeNum = 4
        elif (hourEpisodeNum >= 22) & (hourEpisodeNum < 23):
            episodeNum = 5
        else:
            episodeNum = 0
        return int(episodeNum), int((timeStamp - startTime) // 3600 % 24 + 1),weekdayFlag, int(dayNum)

    hisInputData = [[] for i in range(matrixSize)] # Matrix that records all transitions. 5 * # clusters * # transitions within one certain cluster.
                      # [[episode][cluster][transitions]]

    transitionMatrixDetination = np.zeros((2,5,matrixSize,matrixSize,2)) # Matrix that records transition numbers between 2 certain

    transitionMatrixDuration = [] # Matrix that records all transition durations of two certain regions in a certain episode

    stationStatusMatrix = []

    weatherMatrix = [[0]]

    periodPointer = 8*np.ones(matrixSize)
    dayPointer = np.zeros(matrixSize)

    filename =  transitionDataLoc # transition data
    fRecord = open(filename)
    readerRecord = csv.reader(fRecord)

    filename =  weatherDataLoc # weather data
    fWeather = open(filename)
    readerWeather = csv.reader(fWeather)

    filename = stationStatusDataLoc  # weather data
    fStatus = open(filename)
    readerStatus = csv.reader(fStatus)

    transitionCounter = 0 # counting transition (between different pairs of stations) numbers

    for rowRec in readerRecord:
        [episodeNum,periodNum,weekdayFlag,dayNum] = get_attributes(int(rowRec[loadStartTime])) # initialize
        #print(weekdayFlag,dayNum,episodeNum,periodNum)
        if rowRec[loadEndStationID] != 'NULL':
            if episodeNum != 0:
                stationID = int(rowRec[loadStartStaionID])
                if weekdayFlag == True:
                    weekdayFlagNum = 1
                else:
                    weekdayFlagNum = 0

                if len(hisInputData[stationID]) == 0:
                    while dayNum != dayPointer[stationID]:
                        hisInputData[stationID].append([[], [], [], [], []])
                        while periodPointer[stationID] != 23:
                            hisInputData[stationID][int(dayPointer[stationID])][episodeNum - 1].append(
                                [weekdayFlagNum, 0])
                            periodPointer[stationID] += 1
                        periodPointer[stationID] = 8
                        dayPointer[stationID] += 1
                    if periodPointer[stationID] == periodNum:
                        hisInputData[stationID].append([[[weekdayFlagNum, 1]],[],[],[],[]])
                    else:
                        hisInputData[stationID].append([[],[],[],[],[]])
                        while (periodPointer[stationID] != periodNum):
                            hisInputData[stationID][dayNum][episodeNum-1].append([weekdayFlagNum, 0])
                            periodPointer[stationID] += 1
                        hisInputData[stationID][dayNum][episodeNum-1].append([weekdayFlagNum, 1])

                elif dayNum != dayPointer[stationID]:
                    hisInputData[stationID].append([[], [], [], [], []])
                    if periodPointer[stationID] == 23:
                        periodPointer[stationID] = 8
                        dayPointer[stationID] += 1
                    else:
                        while periodPointer[stationID] != 23:
                            hisInputData[stationID][int(dayPointer[stationID])][episodeNum - 1].append(
                                [weekdayFlagNum, 0])
                            periodPointer[stationID] += 1
                        periodPointer[stationID] = 8
                        dayPointer[stationID] += 1
                    while dayNum != dayPointer[stationID]:
                        hisInputData[stationID].append([[], [], [], [], []])
                        while periodPointer[stationID] != 23:
                            hisInputData[stationID][int(dayPointer[stationID])][episodeNum - 1].append(
                                [weekdayFlagNum, 0])
                            periodPointer[stationID] += 1
                        periodPointer[stationID] = 8
                        dayPointer[stationID] += 1
                    while periodPointer[stationID] != periodNum:
                        hisInputData[stationID][int(dayPointer[stationID])][episodeNum - 1].append([weekdayFlagNum, 0])
                        periodPointer[stationID] += 1
                    hisInputData[stationID][int(dayPointer[stationID])][episodeNum - 1].append([weekdayFlagNum, 1])

                elif periodPointer[stationID] == periodNum:
                    hisInputData[stationID][dayNum][episodeNum-1][-1][1] += 1

                elif periodPointer[stationID] != periodNum:
                    if periodPointer[stationID] < periodNum:
                        periodPointer[stationID] += 1
                        while periodPointer[stationID] != periodNum:
                            hisInputData[stationID][dayNum][episodeNum-1].append([weekdayFlagNum, 0])
                            periodPointer[stationID] += 1
                        hisInputData[stationID][dayNum][episodeNum-1].append([weekdayFlagNum, 1])
                    elif periodPointer[stationID] > periodNum:
                        #print(stationID,periodPointer[stationID],periodNum)
                        hisInputData[stationID][dayNum][episodeNum-1][periodNum-8][1] += 1
                    else:
                        hisInputData[stationID][dayNum][episodeNum-1][-1][1] += 1
                # transition matrix
                transitionMatrixDetination[weekdayFlagNum][episodeNum-1][int(rowRec[loadStartStaionID])][int(rowRec[loadEndStationID])][0] += 1
                #print(transitionMatrixDetination[int(rowRec[3])][int(rowRec[5])])
                if transitionMatrixDetination[weekdayFlagNum][episodeNum-1][int(rowRec[loadStartStaionID])][int(rowRec[loadEndStationID])][0]==1:
                    transitionCounter += 1
                    transitionMatrixDetination[weekdayFlagNum][episodeNum-1][int(rowRec[loadStartStaionID])][int(rowRec[loadEndStationID])][1] = transitionCounter
                    transitionMatrixDuration.append([])
                    transitionMatrixDuration[-1].append(float(int(rowRec[loadDuration])))
                else:
                    transitionMatrixDuration[int(transitionMatrixDetination[weekdayFlagNum][episodeNum-1][int(rowRec[loadStartStaionID])]
                                                 [int(rowRec[loadEndStationID])][1]-1)].append(float(rowRec[loadDuration]))

    weatherDayPointer = 0
    weatherPeriodPointer = 8
    for rowWea in readerWeather:
        [episodeNum, periodNum, _, dayNum] = get_attributes(math.ceil(float(rowWea[0])))
        if episodeNum != 0:
            if dayNum != weatherDayPointer:
                weatherMatrix.append([0])
                weatherDayPointer += 1
                weatherPeriodPointer = 8
            if weatherPeriodPointer != periodNum:
                weatherMatrix[dayNum].append(0)
                weatherPeriodPointer += 1
            if rowWea[2] == '1':
                weatherMatrix[dayNum][periodNum - 8] = 1 #rain
            elif rowWea[3] == '1':
                weatherMatrix[dayNum][periodNum - 8] = 2  # snow
            elif (rowWea[3] != '1')&(rowWea[2] != '1')&(math.ceil(float(rowWea[1]))<=5):
                weatherMatrix[dayNum][periodNum - 8] = 3  # fog
            else:
                weatherMatrix[dayNum][periodNum - 8] = 0  # sunny
    #print(hisInputData)
    #print(weatherMatrix)

   #
    return hisInputData,transitionMatrixDuration,transitionMatrixDetination,weatherMatrix

def calculate_similarity(weatherMatrix,nowAttributes,dayNum,periodNum,hisInputData):
    '''
    :param timeStamp1: time now (period)
    :param timeStamp2: target time (period)
    :param stationID: station(region) ID
    :return: the similarity of period1 and period2 of a certain station
    '''
    weatherSimilarityMatrix = [[1, 0.1, 0.3, 0.1], [0.1, 1, 0.7, 0.3], [0.3, 0.7, 1, 0.3],
                               [0.1, 0.3, 0.3, 1]]  # 0:sunny,1:rain,2:snow,3:fog
    dayNow, periodNow, weekdayFlag = nowAttributes
    if (hisInputData != 0)&(hisInputData != 1):
        if hisInputData[0] != weekdayFlag:
            similarityNum = 0
        else:
            periodDistance = min(abs(periodNum-periodNow),24-abs(periodNow-periodNum))
            dayDistance = abs(dayNum - dayNow)
            timeSimilarity = 0.4**(periodDistance)*0.6**(dayDistance)
            weatherSimilarity = weatherSimilarityMatrix[weatherMatrix[dayNow][periodNow-7]][weatherMatrix[dayNum][periodNum-7]]
            similarityNum = timeSimilarity*weatherSimilarity
        return [similarityNum,hisInputData[1]]
    else:
        if  weekdayFlag != hisInputData:
            similarityNum = 0
        else:
            periodDistance = min(abs(periodNum-periodNow),24-abs(periodNow-periodNum))
            dayDistance = abs(dayNum - dayNow)
            timeSimilarity = 0.4**(periodDistance)*0.6**(dayDistance)
            #print(timeSimilarity)
            weatherSimilarity = weatherSimilarityMatrix[weatherMatrix[dayNow][periodNow-7]][weatherMatrix[dayNum][periodNum-7]]
            similarityNum = timeSimilarity*weatherSimilarity
        return [similarityNum,0]

def get_topKSimilarity(timeStamp,stationID,hisInputData,weatherMatrix):
    '''
    used to find top-k most similar period with time now for a certain station. Here suppose K = 5
    :param timeStamp: # of day, episode, period and time window should be provided
    :param stationID: station(region) ID
    :return: top-k most similar period's timestamp with time now for a certain station
    '''
    def get_nowAttributes(timeStamp):
        hourEpisodeNum = (timeStamp - startTime) % (3600 * 24) // 3600
        if (hourEpisodeNum >= 7) & (hourEpisodeNum < 11):
            episodeNum = 1
        elif (hourEpisodeNum >= 12) & (hourEpisodeNum < 16):
            episodeNum = 2
        elif (hourEpisodeNum >= 17) & (hourEpisodeNum < 22):
            episodeNum = 3
        elif ((hourEpisodeNum >= 11) & (hourEpisodeNum < 12)) | (hourEpisodeNum >= 16) & (hourEpisodeNum < 17):
            episodeNum = 4
        elif (hourEpisodeNum >= 22) & (hourEpisodeNum < 23):
            episodeNum = 5
        else:
            episodeNum = 0
        dayNow = ((timeStamp - startTime) // (3600 * 24))
        if (dayNow % 7 == 5) | (dayNow % 7 == 6):
            weekdayFlag = 0
        else:
            weekdayFlag = 1
        periodNow = int((timeStamp - startTime) // 3600 % 24)
        return dayNow,periodNow,weekdayFlag,episodeNum
    similarity = []
    dayNow, periodNow, weekdayFlag, episodeNum = get_nowAttributes(timeStamp)
    nowAttributes = [dayNow, periodNow, weekdayFlag]
    for dayNum in range(dayNow):
        for periodNum in range(7,7 + 16):
            if (periodNum >= 7) & (periodNum < 11):
                episodeNumTemp = 1
                periodNumTemp = periodNum - 7
            elif (periodNum >= 12) & (periodNum < 16):
                episodeNumTemp = 2
                periodNumTemp = periodNum - 12
            elif (periodNum >= 17) & (periodNum < 22):
                episodeNumTemp = 3
                periodNumTemp = periodNum - 17
            elif (periodNum >= 11) & (periodNum < 12):
                episodeNumTemp = 4
                periodNumTemp = periodNum - 11
            elif (periodNum >= 16) & (periodNum < 17):
                episodeNumTemp = 4
                periodNumTemp = periodNum - 16
            elif (periodNum >= 22) & (periodNum < 23):
                episodeNumTemp = 5
                periodNumTemp = periodNum - 22
            #print(hisInputData[stationID],dayNum)
            if len(hisInputData[stationID][dayNum][episodeNumTemp-1])<periodNumTemp+1:
                similarity.append(calculate_similarity(weatherMatrix,nowAttributes,dayNum,periodNum,hisInputData[stationID][dayNum][0][0][0]))
            else:
                similarity.append(calculate_similarity(weatherMatrix,nowAttributes,dayNum,periodNum,hisInputData[stationID][dayNum][episodeNumTemp-1][periodNumTemp]))
    topKPeriod = sorted(similarity,reverse=True,key=lambda x:x[0])
    sumSimilarity = 0
    sumFrequency = 0
    for i in range(100):
        sumSimilarity += topKPeriod[i][0]
        sumFrequency += topKPeriod[i][1]*topKPeriod[i][0]
    expectedDepartureNumber = int(sumFrequency // sumSimilarity)
    #print(expectedDepartureNumber)
    return expectedDepartureNumber

def IModel(timeStamp,stationID,durationFlag,destinationIDIn,transitionMatrixDuration,transitionMatrixDetination):
    '''
    :param timeStamp: now time
    :param expectedDepartureNumber: expected departure bikes
    :param transitionMatrixDuration: obtained by read-inData(), used to predict duration of a transition
    :param transitionMatrixDetination: obtained by read-inData(), ued to predict destination of a transition
    :param stationID: the station we are considering now
    :return: predicted transitions including destination and duration in the whole period that timeStamp is in
    '''
    def get_episodeNum(timeStamp):
        hourEpisodeNum = (timeStamp - startTime) % (3600 * 24) // 3600
        if (hourEpisodeNum >= 7) & (hourEpisodeNum < 11):
            episodeNum = 1
        elif (hourEpisodeNum >= 12) & (hourEpisodeNum < 16):
            episodeNum = 2
        elif (hourEpisodeNum >= 17) & (hourEpisodeNum < 22):
            episodeNum = 3
        elif ((hourEpisodeNum >= 11) & (hourEpisodeNum < 12)) | (hourEpisodeNum >= 16) & (hourEpisodeNum < 17):
            episodeNum = 4
        elif (hourEpisodeNum >= 22) & (hourEpisodeNum < 23):
            episodeNum = 5
        else:
            episodeNum = 0
        return episodeNum
    dayNum = ((timeStamp - startTime) // (3600 * 24))
    if (dayNum % 7 == 5) | (dayNum % 7 == 6):
        weekdayFlagNum = 0
    else:
        weekdayFlagNum = 1
    expectedTransition = []
    episodeNum = get_episodeNum(timeStamp)
    destinationPredict = []
    for j in range(matrixSize):
        if transitionMatrixDetination[weekdayFlagNum][episodeNum-1][stationID][j][0]!=0:
            for k in range(int(transitionMatrixDetination[weekdayFlagNum][episodeNum-1][stationID][j][0])):
                destinationPredict.append(j)

    destinationID = destinationPredict[random.randint(0,len(destinationPredict)-1)]
    if durationFlag==False:
        return destinationID
    else:
        durationList = transitionMatrixDuration[int(transitionMatrixDetination[weekdayFlagNum][episodeNum-1][stationID]
                                                                [destinationIDIn][1]-1)]
        durationPrediction = durationList[random.randint(0,len(durationList)-1)]
        return durationPrediction

def OModel(timeStamp,stationID,hisInputData,weatherMatrix):
    '''
    OModel
    :param timeStamp: # of day, episode, period and time window should be provided
    :param stationID: station(region) ID
    :return: rent events in this time window (with destination and corresponding during info)
    '''
    expectedDepartureNumber = get_topKSimilarity(timeStamp,stationID,hisInputData,weatherMatrix) # 0:sunny,1:rain,2:snow,3:fog
    expectedDepartureTime = timeStamp - timeStamp % 3600 + random.randint(0,59) * 60
    return expectedDepartureNumber,expectedDepartureTime

def get_expectDepartureNumber(timeStamp,stationID):
    expectedDepartureNumber = OModel(timeStamp,stationID, hisInputDataGlobal, weatherMatrixGlobal)
    return expectedDepartureNumber

def get_predictedDestination(timeStamp,stationID):
    predictedDestination = IModel(timeStamp,stationID, False, 0, transitionMatrixDurationGlobal
                                  , transitionMatrixDetinationGlobal)
    return predictedDestination

def get_predictedDuration(timeStamp,stationID,destinationID):
    predictedDuration = IModel(timeStamp,stationID, True, destinationID,
                               transitionMatrixDurationGlobal, transitionMatrixDetinationGlobal)
    return predictedDuration

'''
Input the timeStamp and station ID, then you will get predicted transitions including destination and duration
'''
time1 = time.time()
hisInputDataGlobal, transitionMatrixDurationGlobal, transitionMatrixDetinationGlobal, weatherMatrixGlobal = read_inData()

expectedDepartureNumber,expectedDepartureTime = get_expectDepartureNumber(1373964540, 212)
predictedDestination = get_predictedDestination(1373964540, 212)
predictedDuration = get_predictedDuration(1373964540, 212, predictedDestination)
time2 = time.time()

print('Time used:',time2-time1,'s')
print(expectedDepartureNumber,expectedDepartureTime)
print(predictedDestination)
print(predictedDuration)

