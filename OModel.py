import numpy as np
import csv
import datetime
import time
import math
import random

'''
Define 5 episode
episode1 = [7,11]
episode2 = [12,16]
episode3 = [17,22]
episode4 = [[11,12],[16,17]]
episode5 = [22,23]'''

'''Loading files'''
weatherDataLoc = 'C:/Users/Tony Xu/OneDrive/KDD Pro/temps.csv'
transitionDataLoc = 'C:/Users/Tony Xu/OneDrive/KDD Pro/test.csv'

'''!!!Do not modify this part!!!'''
weatherSimilarityMatrix = [[1,0.1,0.3,0.1],[0.1,1,0.7,0.3],[0.3,0.7,1,0.3],[0.1,0.3,0.3,1]] # 0:sunny,1:rain,2:snow,3:fog

def get_timeStamp(timeIn):
    date_time = datetime.datetime.strptime(timeIn, '%Y/%m/%d %H:%M')
    time_time = time.mktime(date_time.timetuple())
    return time_time

def get_timeString(timeStamp):
    nowTime = datetime.datetime.fromtimestamp(timeStamp)
    timeString = nowTime.strftime("%Y/%m/%d %H:%M:%S")
    return timeString

def read_inData():
    '''
    read in data and do preprocessing.
    :return:
    hisInputData: Get departure data of each station in every period (including 0 departure)
    transitionMatrixDuration: Get transition matrix recording # transitions between two certain stations
        5*4000*4000*2 matrix:[epsioderNum,start station ID, end station ID, [frequent,corresponding ID in
        transitionMatrixDetination]]
    transitionMatrixDetination: Get trasition duration lists recording all duratin time between two certain stations
        n*m matrix:[transition ID(recorded in transitionMatrixDuration),[all transition duration of two stations]]
    '''
    def get_weekFalg(timeInWeekFlag):
        timeStamp = get_timeStamp(timeInWeekFlag)
        dayNum = ((timeStamp - 1371139200) // (3600 * 24))
        if (dayNum % 7 == 1) | (dayNum % 7 == 2):
            weekdayFlag = False
        else:
            weekdayFlag = True
        return weekdayFlag, int(dayNum)

    def get_episodeNum(timeInEpisodeNum):
        timeStamp = get_timeStamp(timeInEpisodeNum)
        hourEpisodeNum = (timeStamp - 1371139200) % (3600 * 24) // 3600
        #print(hourEpisodeNum)
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
        return episodeNum, int((timeStamp - 1371139200) // 3600 % 24 + 1)

    hisInputData = []
    transitionMatrixDetination = np.zeros((5,4000,4000,2))
    transitionMatrixDuration = []
    periodPointer = 8
    dayPointer = 0

    filename =  transitionDataLoc # transition data
    fRecord = open(filename)
    readerRecord = csv.reader(fRecord)

    filename =  weatherDataLoc # weather data
    fWeather = open(filename)
    readerWeather = csv.reader(fWeather)

    transitionCounter = 0
    for rowRec in readerRecord:
        [[weekdayFlag,dayNum], [episodeNum, periodNum]] = [get_weekFalg(rowRec[1]), get_episodeNum(rowRec[1])]
        #print(weekdayFlag,dayNum,episodeNum,periodNum)
        if rowRec[5] != 'NULL':
            if episodeNum != 0:
                if weekdayFlag == True:
                    weekdayFlagNum = 1
                else:
                    weekdayFlagNum = 0

                if len(hisInputData) == 0:
                    if periodPointer == periodNum:
                        hisInputData.append([[weekdayFlagNum, 1]])
                    else:
                        hisInputData.append([])
                        while (periodPointer != periodNum):
                            hisInputData[dayNum].append([weekdayFlagNum, 0])
                            periodPointer += 1
                        hisInputData[dayNum].append([weekdayFlagNum, 1])
                elif periodPointer == periodNum:
                    hisInputData[dayNum][-1][1] += 1
                elif periodPointer != periodNum:
                    if dayNum != dayPointer:
                        hisInputData.append([])
                        if periodPointer == 23:
                            periodPointer = 8
                            dayPointer += 1
                            hisInputData[dayPointer].append([weekdayFlagNum,0])
                        else:
                            while periodPointer != 23:
                                hisInputData[dayPointer].append([weekdayFlagNum, 0])
                                periodPointer += 1
                            periodPointer = 8
                            dayPointer += 1
                            hisInputData[dayPointer].append([weekdayFlagNum, 0])
                    if periodPointer < periodNum:
                        periodPointer += 1
                        while periodPointer != periodNum:
                            hisInputData[dayNum].append([weekdayFlagNum, 0])
                            periodPointer += 1
                        hisInputData[dayNum].append([weekdayFlagNum, 1])
                    elif periodPointer > periodNum:
                        #print(weekdayFlag, dayNum, episodeNum, periodNum)
                        hisInputData[dayNum][periodNum-7][1] += 1
                    else:
                        hisInputData[dayNum][-1][1] += 1

            # transition matrix
            transitionMatrixDetination[episodeNum-1][int(rowRec[3])][int(rowRec[5])][0] += 1
            #print(transitionMatrixDetination[int(rowRec[3])][int(rowRec[5])])
            if transitionMatrixDetination[episodeNum-1][int(rowRec[3])][int(rowRec[5])][0]==1:
                transitionCounter += 1
                transitionMatrixDetination[episodeNum-1][int(rowRec[3])][int(rowRec[5])][1] = transitionCounter
                transitionMatrixDuration.append([])
                transitionMatrixDuration[-1].append(float(int(rowRec[0])))
            else:
                transitionMatrixDuration[int(transitionMatrixDetination[episodeNum-1][int(rowRec[3])][int(rowRec[5])][1]-1)].append(float(rowRec[0]))
    #print(hisInputData)

    for rowWea in readerWeather:
        [[weekdayFlag,dayNum], [episodeNum, periodNum]] = [get_weekFalg(rowWea[0]), get_episodeNum(rowWea[0])]
        if episodeNum != 0:
            if len(hisInputData[dayNum][periodNum-8]) == 2:
                if rowWea[2] == '1':
                    hisInputData[dayNum][periodNum-8].append(1) #rain
                elif rowWea[3] == '1':
                    hisInputData[dayNum][periodNum-8].append(2)  # snow
                elif (rowWea[3] != '1')&(rowWea[2] != '1')&(math.ceil(int(rowWea[1]))<=5):
                    hisInputData[dayNum][periodNum-8].append(3)  # fog
                else:
                    hisInputData[dayNum][periodNum-8].append(0)  # sunny
    #print(hisInputData)
    return hisInputData,transitionMatrixDuration,transitionMatrixDetination

def calculate_similarity(nowAttributes,dayNum,periodNum,hisInputData):
    '''
    :param timeStamp1: time now (period)
    :param timeStamp2: target time (period)
    :param stationID: station(region) ID
    :return: the similarity of period1 and period2 of a certain station
    '''
    dayNow, periodNow, weekdayFlag,weatherPrediction = nowAttributes
    similarityNum = 0
    timeSimilarity = 0
    weatherSimilarity = 0
    if hisInputData[0] != weekdayFlag:
        similarityNum = 0
    else:
        periodDistance = min(abs(periodNum-periodNow),24-abs(periodNow-periodNum))
        dayDistance = abs(dayNum - dayNow)
        timeSimilarity = 0.4**(periodDistance)*0.6**(dayDistance)
        #print(timeSimilarity)
        weatherSimilarity = weatherSimilarityMatrix[weatherPrediction][hisInputData[2]]
        similarityNum = timeSimilarity*weatherSimilarity
    return [similarityNum,hisInputData[1]]

def get_topKSimilarity(timeStamp,stationID,hisInputData,weatherPrediction):
    '''
    used to find top-k most similar period with time now for a certain station. Here suppose K = 5
    :param timeStamp: # of day, episode, period and time window should be provided
    :param stationID: station(region) ID
    :return: top-k most similar period's timestamp with time now for a certain station
    '''
    def get_nowAttributes(timeStamp):
        dayNow = ((timeStamp - 1371139200) // (3600 * 24))
        if (dayNow % 7 == 1) | (dayNow % 7 == 2):
            weekdayFlag = 0
        else:
            weekdayFlag = 1
        periodNow = int((timeStamp - 1371139200) // 3600 % 24)
        return dayNow,periodNow,weekdayFlag
    similarity = []
    dayNow, periodNow, weekdayFlag = get_nowAttributes(timeStamp)
    nowAttributes = [dayNow, periodNow, weekdayFlag,weatherPrediction]
    topKPeriod = []
    for dayNum in range(dayNow):
        for periodNum in range(16):
            similarity.append(calculate_similarity(nowAttributes,dayNum,periodNum,hisInputData[dayNum][periodNum]))
    topKPeriod = sorted(similarity,reverse=True,key=lambda x:x[0])
    sumSimilarity = 0
    sumFrequency = 0
    for i in range(100):
        sumSimilarity += topKPeriod[i][0]
        sumFrequency += topKPeriod[i][1]*topKPeriod[i][0]
    expectedDepartureNumber = int(2 * sumFrequency // sumSimilarity)
    #print(expectedDepartureNumber)
    return expectedDepartureNumber

def IModel(timeStamp,expectedDepartureNumber,transitionMatrixDuration,transitionMatrixDetination,stationID):
    '''
    :param timeStamp: now time
    :param expectedDepartureNumber: expected departure bikes
    :param transitionMatrixDuration: obtained by read-inData(), used to predict duration of a transition
    :param transitionMatrixDetination: obtained by read-inData(), ued to predict destination of a transition
    :param stationID: the station we are considering now
    :return: predicted transitions including destination and duration in the whole period that timeStamp is in
    '''
    def get_episodeNum(timeStamp):
        hourEpisodeNum = (timeStamp - 1371139200) % (3600 * 24) // 3600
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
    expectedTransition = []
    episodeNum = get_episodeNum(timeStamp)
    destinationPredict = []
    for j in range(4000):
        if transitionMatrixDetination[episodeNum-1][stationID][j][0]!=0:
            for k in range(int(transitionMatrixDetination[episodeNum-1][stationID][j][0])):
                destinationPredict.append(j)
    for i in range(expectedDepartureNumber):
        timeWindowNum = random.randint(0,59)
        destinationID = destinationPredict[random.randint(0,len(destinationPredict)-1)]
        durationList = transitionMatrixDuration[int(transitionMatrixDetination[episodeNum-1][stationID]
                                                                  [destinationID][1]-1)]
        #print(durationList)
        durationPrediction = durationList[random.randint(0,len(durationList)-1)]
        expectedTransition.append([get_timeString(timeStamp+timeWindowNum*60),timeStamp+timeWindowNum*60,destinationID,
                                   durationPrediction])

    outputExpectedTransition = sorted(expectedTransition,key=lambda x:x[1])
    return outputExpectedTransition

def OModel(timeStamp,stationID):
    '''
    OModel
    :param timeStamp: # of day, episode, period and time window should be provided
    :param stationID: station(region) ID
    :return: rent events in this time window (with destination and corresponding during info)
    '''
    ongoingTransitions = []
    hisInputData,transitionMatrixDuration,transitionMatrixDetination = read_inData()
    expectedDepartureNumber = get_topKSimilarity(timeStamp,stationID,hisInputData,0) # 0:sunny,1:rain,2:snow,3:fog
    predictions = IModel(timeStamp,expectedDepartureNumber,transitionMatrixDuration,transitionMatrixDetination,stationID)
    print('Predicted departure number in this period:',expectedDepartureNumber)
    for pre in predictions:
        print('Predictioned transition:',pre)
    return predictions

'''
Input the timeStamp and station ID, then you will get predicted transitions including destination and duration
'''
time1 = time.time()
predictedTransitions = OModel(1372410000,476) # since I only use station 476 for testing, do no change station ID here
time2 = time.time()
print('Time used:',time2-time1,'s')