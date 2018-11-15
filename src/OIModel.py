import numpy as np
import csv
import datetime
import time
import math
import random
import matplotlib.pyplot as plt
import xgboost as xgb

'''
Define 5 episode
episode0 = [7,11]
episode2 = [12,16]
episode3 = [17,18]
episode1 = [[11,12],[16,17]]
episode4 = [18,23]'''


startTime = 1375286400
matrixSize = 33

def get_timeString(timeStamp):
    nowTime = datetime.datetime.fromtimestamp(timeStamp)
    timeString = nowTime.strftime("%Y/%m/%d %H:%M:%S")
    return timeString

def read_inData(transitionDataLoc,weatherDataLoc):
    '''

    '''

    '''Initialize'''

    [loadDuration, loadStartTime, loadEndTime, loadStartStaionID, loadEndStationID, loadClusterTag] = [0, 7, 2, 10, 11, 6]

    '''!!!Do not modify this part!!!'''
    def get_attributes(timeStamp):
        '''
        return True if yes, False if no
        return # day caompared with start time
        return episode number
        return period number compared with start time
        '''
        dayNum = ((timeStamp - startTime) // (3600 * 24))
        periodNum = int((timeStamp - startTime) // 3600 % 24)
        if (dayNum % 7 == 3) | (dayNum % 7 == 2):
            weekdayFlag = False
        else:
            weekdayFlag = True
        hourEpisodeNum = (timeStamp - startTime) % (3600 * 24) // 3600
        if (hourEpisodeNum >= 7) & (hourEpisodeNum < 11):
            episodeNum = 0
            periodNum -= 7
        elif (hourEpisodeNum >= 12) & (hourEpisodeNum < 16):
            episodeNum = 2
            periodNum -= 12
        elif (hourEpisodeNum >= 17) & (hourEpisodeNum < 18):
            episodeNum = 3
            periodNum -= 17
        elif (hourEpisodeNum >= 11) & (hourEpisodeNum < 12):
            episodeNum = 1
            periodNum -= 11
        elif (hourEpisodeNum >= 16) & (hourEpisodeNum < 17):
            episodeNum = 1
            periodNum -= 17
        elif (hourEpisodeNum >= 18) & (hourEpisodeNum < 23):
            episodeNum = 4
            periodNum -= 18
        else:
            episodeNum = -1
        return int(episodeNum),periodNum ,weekdayFlag, int(dayNum)

    hisInputData = np.zeros((5,matrixSize,65,5)) # Matrix that records all transitions. 5 * # clusters * # transitions within one certain cluster.
                      # [[episode][cluster][transitions]]

    transitionMatrixDetination = np.zeros((2,5,matrixSize,matrixSize,2)) # Matrix that records transition numbers between 2 certain

    transitionMatrixDuration = [] # Matrix that records all transition durations of two certain regions in a certain episode

    weatherMatrix = np.zeros((65,16))

    weatherMatrixAll = np.zeros((65,16,4))

    periodPointer = 8*np.ones(matrixSize)
    dayPointer = np.zeros(matrixSize)

    filename =  transitionDataLoc # transition data
    fRecord = open(filename)
    readerRecord = csv.reader(fRecord)

    filename =  weatherDataLoc # weather data
    fWeather = open(filename)
    readerWeather = csv.reader(fWeather)



    transitionCounter = 0 # counting transition (between different pairs of stations) numbers

    for rowRec in readerRecord:
        if rowRec[0] != 'tripduration':
            [episodeNum,periodNum,weekdayFlag,dayNum] = get_attributes(int(rowRec[loadStartTime])+1372608000) # initialize
            #print(weekdayFlag,dayNum,episodeNum,periodNum)
            if int(rowRec[loadEndStationID]) <= 33:
                stationID = int(rowRec[loadStartStaionID])
                if weekdayFlag == True:
                    weekdayFlagNum = 1
                else:
                    weekdayFlagNum = 0

                hisInputData[episodeNum][stationID][dayNum][periodNum] += 1

                    # transition matrix
                transitionMatrixDetination[weekdayFlagNum][episodeNum][int(rowRec[loadStartStaionID])][int(rowRec[loadEndStationID])][0] += 1
                    #print(transitionMatrixDetination[int(rowRec[3])][int(rowRec[5])])
                if transitionMatrixDetination[weekdayFlagNum][episodeNum][int(rowRec[loadStartStaionID])][int(rowRec[loadEndStationID])][0]==1:
                    transitionCounter += 1
                    transitionMatrixDetination[weekdayFlagNum][episodeNum][int(rowRec[loadStartStaionID])][int(rowRec[loadEndStationID])][1] = transitionCounter
                    transitionMatrixDuration.append([])
                    transitionMatrixDuration[-1].append(float(int(rowRec[loadDuration])))
                else:
                    transitionMatrixDuration[int(transitionMatrixDetination[weekdayFlagNum][episodeNum][int(rowRec[loadStartStaionID])]
                                                     [int(rowRec[loadEndStationID])][1]-1)].append(float(rowRec[loadDuration]))

    #print(hisInputData[1][21][0])

    def get_attributesWeather(timeStamp):
        dayNum = ((timeStamp-startTime) // (3600 * 24))
        return int((timeStamp - startTime) // 3600 % 24 + 1), int(dayNum)

    for rowWea in readerWeather:
        if rowWea[0]!= 'datetime':
            if (rowWea[3] != 'NA') & (rowWea[2] != 'NA')&(rowWea[1] != 'NA')&(rowWea[4] != 'NA')&(rowWea[5] != 'NA'):
                [periodNum, dayNum] = get_attributesWeather(math.ceil(float(rowWea[0])))
                if (periodNum>=8)&(periodNum<23):
                    if rowWea[5] == '1':
                        weatherMatrix[dayNum][periodNum - 8] = 1 #rain
                        weatherMatrixAll[dayNum][periodNum - 8] = [float(rowWea[1]),float(rowWea[2]),float(rowWea[3]),1]
                    elif rowWea[6] == '1':
                        weatherMatrix[dayNum][periodNum - 8] = 2 # snow
                        weatherMatrixAll[dayNum][periodNum - 8] = [float(rowWea[1]), float(rowWea[2]),float(rowWea[3]),2]
                    elif (rowWea[6] != '1')&(rowWea[5] != '1')&(math.ceil(float(rowWea[4]))<=5):
                        weatherMatrix[dayNum][periodNum - 8] = 3  # fog
                        weatherMatrixAll[dayNum][periodNum - 8] = [float(rowWea[1]),float(rowWea[2]), float(rowWea[3]),3]
                    else:
                        weatherMatrix[dayNum][periodNum - 8] = 0  # sunny
                        weatherMatrixAll[dayNum][periodNum - 8] = [float(rowWea[1]),  float(rowWea[2]),float(rowWea[3]),0]
    #print(hisInputData)
    #print(weatherMatrix)

    entireSituation = np.zeros((62,16))
    for i in range(62):
        for j in range(16):
            if (j+7 >= 7) & (j+7 < 11):
                episodeNum = 0
                periodNum = j+7 - 7
            elif (j+7 >= 12) & (j+7 < 16):
                episodeNum = 2
                periodNum = j + 7 - 12
            elif (j+7 >= 17) & (j+7 < 18):
                episodeNum = 3
                periodNum = j + 7 - 17
            elif (j+7 >= 11) & (j+7 < 12):
                episodeNum = 1
                periodNum = j + 7 - 11
            elif (j+7 >= 16) & (j+7 < 17):
                episodeNum = 1
                periodNum = j + 7 - 17
            elif (j+7 >= 18) & (j+7 < 23):
                episodeNum = 4
                periodNum = j + 7 - 18
            else:
                episodeNum = -1
            for k in range(matrixSize):
                entireSituation[i][j] += hisInputData[episodeNum][k][i][periodNum]
    #print(entireSituation)
    return hisInputData,transitionMatrixDuration,transitionMatrixDetination,weatherMatrix,entireSituation,weatherMatrixAll


def calculate_similarity(weatherMatrix,nowAttributes,dayNum,periodNum):
    '''
    :param timeStamp1: time now (period)
    :param timeStamp2: target time (period)
    :param stationID: station(region) ID
    :return: the similarity of period1 and period2 of a certain station
    '''
    def weatherSame(timeStamp,dayNum,dayNow):
        dayNum1 = ((timeStamp - startTime) // (3600 * 24))
        timeStamp2 = timeStamp + dayNum - dayNow
        dayNum2 = ((timeStamp2 - startTime) // (3600 * 24))
        if (dayNum1 % 7 == 3) | (dayNum1 % 7 == 2):
            weekdayFlag1 = False
        else:
            weekdayFlag1 = True

        if (dayNum2 % 7 == 2) | (dayNum2 % 7 == 3):
            weekdayFlag2= False
        else:
            weekdayFlag2 = True
        if weekdayFlag1 == weekdayFlag2:
            return 1
        else:
            return 0

    weatherSimilarityMatrix = [[1, 0.1, 0.05, 0.1],
                               [0.1, 1, 0.3, 0.4],
                               [0.05, 0.3, 1, 0.2],
                               [0.1, 0.4, 0.2, 1]]  # 0:sunny,1:rain,2:snow,3:fog
    dayNow, periodNow, weekdayFlag,timeStamp = nowAttributes
    if weatherSame(timeStamp,dayNum,dayNow)==0:
        similarityNum = 0
    else:
        periodDistance = min(abs(periodNum-periodNow),24-abs(periodNow-periodNum))
        dayDistance = abs(dayNum - dayNow)
        weatherSimilarity = weatherSimilarityMatrix[int(weatherMatrix[int(dayNow)][int(periodNow)-7])][int(weatherMatrix[int(dayNum)][int(periodNum)-7])]
        timeSimilarity = 0.5 ** (periodDistance) *  0.3 ** (dayDistance)
        similarityNum = timeSimilarity*weatherSimilarity
    return similarityNum

def get_topKSimilarity(timeStamp,stationID,hisInputData,weatherMatrix,overAllTest,pred_y,dayNumBound):
    '''
    used to find top-k most similar period with time now for a certain station. Here suppose K = 5
    :param timeStamp: # of day, episode, period and time window should be provided
    :param stationID: station(region) ID
    :return: top-k most similar period's timestamp with time now for a certain station
    '''
    def get_nowAttributes(timeStamp):
        hourEpisodeNum = (timeStamp - startTime) % (3600 * 24) // 3600
        if (hourEpisodeNum >= 7) & (hourEpisodeNum < 11):
            episodeNum = 0
            periodStrat = 7
            delta = 4
        elif (hourEpisodeNum >= 12) & (hourEpisodeNum < 16):
            episodeNum = 2
            periodStrat = 12
            delta = 4
        elif (hourEpisodeNum >= 17) & (hourEpisodeNum < 18):
            episodeNum = 3
            periodStrat = 17
            delta = 1
        elif ((hourEpisodeNum >= 11) & (hourEpisodeNum < 12)) | (hourEpisodeNum >= 16) & (hourEpisodeNum < 17):
            episodeNum = 1
            periodStrat = 11
            delta = 5
        elif (hourEpisodeNum >= 18) & (hourEpisodeNum < 23):
            episodeNum = 4
            periodStrat = 18
            delta = 5
        else:
            episodeNum = -1
        dayNow = ((timeStamp - startTime) // (3600 * 24))
        if (dayNow % 7 == 3) | (dayNow % 7 == 2):
            weekdayFlag = 0
        else:
            weekdayFlag = 1
        periodNow = int((timeStamp - startTime) // 3600 % 24)
        return dayNow,periodNow,weekdayFlag,episodeNum,periodStrat,delta

    similarity = []
    dayNow, periodNow, weekdayFlag, episodeNum, periodStart,delta = get_nowAttributes(timeStamp)
    nowAttributes = [dayNow, periodNow, weekdayFlag,timeStamp]
    for dayNum in range(10):
        for periodNum in range(periodStart,periodStart+delta):
            if hisInputData[episodeNum][stationID][dayNum][periodNum-periodStart]!=0:
                similarity.append([calculate_similarity(weatherMatrix,nowAttributes,dayNum,periodNum)
                    ,hisInputData[episodeNum][stationID][dayNum][periodNum-periodStart]/overAllTest[dayNum*16+periodNum-7]])

    topKPeriod = sorted(similarity,reverse=True,key=lambda x:x[0])
    sumSimilarity = 0
    sumFrequency = 0
    for i in range(10):
        sumSimilarity += topKPeriod[i][0]
        sumFrequency += topKPeriod[i][1]*topKPeriod[i][0]
    expectedDepartureNumber = int(sumFrequency / sumSimilarity * pred_y[(dayNow-dayNumBound)*16+periodNow-7])
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
    destinationOut = []

    def get_episodeNum(timeStamp):
        hourEpisodeNum = (timeStamp - startTime) % (3600 * 24) // 3600
        if (hourEpisodeNum >= 7) & (hourEpisodeNum < 11):
            episodeNum = 0
        elif (hourEpisodeNum >= 12) & (hourEpisodeNum < 16):
            episodeNum = 2
        elif (hourEpisodeNum >= 17) & (hourEpisodeNum < 18):
            episodeNum = 3
        elif ((hourEpisodeNum >= 11) & (hourEpisodeNum < 12)) | (hourEpisodeNum >= 16) & (hourEpisodeNum < 17):
            episodeNum = 1
        elif (hourEpisodeNum >= 18) & (hourEpisodeNum < 23):
            episodeNum = 4
        else:
            episodeNum = -1
        return episodeNum
    dayNum = ((timeStamp - startTime) // (3600 * 24))
    if (dayNum % 7 ==2) | (dayNum % 7 ==3):
        weekdayFlagNum = 0
    else:
        weekdayFlagNum = 1
    expectedTransition = []
    episodeNum = get_episodeNum(timeStamp)
    destinationPredict = []
    destinationID = []
    for j in range(matrixSize):
        if transitionMatrixDetination[weekdayFlagNum][episodeNum][stationID][j][0] != 0:
            destinationPredict.append(int(transitionMatrixDetination[weekdayFlagNum][episodeNum][stationID][j][0]))
            destinationID.append(j)
    predictedDestions = np.random.choice(destinationID, 1, p=np.array(destinationPredict) / sum(destinationPredict))

    if durationFlag==False:
        return predictedDestions
    else:
        durationList = transitionMatrixDuration[int(transitionMatrixDetination[weekdayFlagNum][episodeNum][stationID]
                                                                [destinationIDIn][1]-1)]
        durationPrediction = durationList[random.randint(0,len(durationList)-1)]
        return durationPrediction

def OModel(timeStamp,stationID,hisInputData,weatherMatrix,overAllTest,pred_y,dayNumBound):
    '''
    OModel
    :param timeStamp: # of day, episode, period and time window should be provided
    :param stationID: station(region) ID
    :return: rent events in this time window (with destination and corresponding during info)
    '''
    strTime = []
    expectedDepartureNumber = get_topKSimilarity(timeStamp,stationID,hisInputData,weatherMatrix,overAllTest,pred_y,dayNumBound) # 0:sunny,1:rain,2:snow,3:fog
    for i in range(expectedDepartureNumber):
        predictedTime = (timeStamp - timeStamp % 3600 + random.randint(0, 59) * 60)
        strTime.append(predictedTime)
    return strTime

def get_expectDepartureNumber(timeStamp,stationID,hisInputDataGlobal, weatherMatrixGlobal,overAllTest,pret_y,timeBound):
    expectedDepartureNumber = OModel(timeStamp,stationID, hisInputDataGlobal, weatherMatrixGlobal,overAllTest,pret_y,timeBound)
    return expectedDepartureNumber

def get_predictedDestination(timeStamp,stationID,transitionMatrixDurationGlobal
                                  , transitionMatrixDetinationGlobal):
    predictedDestination = IModel(timeStamp,stationID, False, 0, transitionMatrixDurationGlobal
                                  , transitionMatrixDetinationGlobal)
    return predictedDestination

def get_predictedDuration(timeStamp,stationID,destinationID,transitionMatrixDurationGlobal, transitionMatrixDetinationGlobal):
    predictedDuration= IModel(timeStamp,stationID, True, destinationID,
                               transitionMatrixDurationGlobal, transitionMatrixDetinationGlobal)
    return predictedDuration

def predicted_overallSituation(dayNumBound,entireSituation,weatherMatrixAll):
    def overall_sit():
        overAllTrain = []
        overAllTest = []
        testData = []
        testReal = []
        for i in range(dayNumBound):
            for j in range(16):
                if (i%7==2)&(i%7==3):
                    weekdayFlag = 0
                else:
                    weekdayFlag = 1
                overAllTrain.append([j,weatherMatrixAll[i][j][0],weatherMatrixAll[i][j][1],weatherMatrixAll[i][j][2],
                                     weatherMatrixAll[i][j][3],weekdayFlag])
                overAllTest.append(entireSituation[i][j])
        for i in range(dayNumBound,61):
            for j in range(16):
                testData.append([j,weatherMatrixAll[i][j][0],weatherMatrixAll[i][j][1],weatherMatrixAll[i][j][2],
                                     weatherMatrixAll[i][j][3],weekdayFlag])
                testReal.append(entireSituation[i][j])
        return overAllTrain,overAllTest,testData,testReal



    overAllTrain,overAllTest,testData,testReal = overall_sit()
    dtrain = xgb.DMatrix(overAllTrain, overAllTest)
    dtest = xgb.DMatrix(testData)
    xgb_params = {
        'eta': 0.01,
        'max_depth': 7,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'silent': 1
    }
    model = xgb.train(xgb_params, dtrain, num_boost_round=5000)
    pred_y = model.predict(dtest)
    '''plt.plot(testReal)
    plt.plot(pred_y)
    plt.show()'''
    print (sum(abs(testReal-pred_y))/sum(testReal))
    return overAllTest, pred_y, dayNumBound
''' '''
