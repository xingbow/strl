import csv
import datetime
import matplotlib.pyplot as plt
import OIModel as om
import time

time1 = time.time()
# type in file loc
transitionDataLoc = 'C:/Users/Tony Xu/OneDrive/KDD 2/test.csv' # transition file
weatherDataLoc = 'C:/Users/Tony Xu/OneDrive/KDD 2/weather.csv' # weather info file

# Initialize function, no need to modify
hisInputData, transitionMatrixDuration, transitionMatrixDetination, weatherMatrix = om.read_inData(transitionDataLoc,weatherDataLoc)

# Inputs time stamp : e.g. 1373964540
# Inputs station ID : e.g.212
# return: expected departure(rent) number of this station in a period(hour)
timeStamp = 18710507+1372608000
startStationID = 11
expectedDepartureNumber = om.get_expectDepartureNumber(timeStamp,startStationID,hisInputData,weatherMatrix)

# Inputs time stamp : e.g. 1373964540
# Inputs station ID : e.g.212
# return: expected destination and departure time
# return: reference result is the same as predictedDestination except that it uses string time, thus more intuitive
timeStamp = 18710507+1372608000
startStationID = 11
predictedDestination,referenceResult = om.get_predictedDestination(timeStamp,startStationID,transitionMatrixDuration, transitionMatrixDetination,expectedDepartureNumber)

# Inputs time stamp : e.g. 1373964540
# Inputs start station ID : e.g.212
# Inputs end station ID : e.g.404
# return: expected time used
timeStamp = 18710507+1372608000
startStationID = 11
endStationID = predictedDestination[0][0]
predictedDuration = om.get_predictedDuration(timeStamp,startStationID,endStationID ,transitionMatrixDuration, transitionMatrixDetination)
time2 = time.time()

print('Expected departure numbers of station',startStationID,'is:',expectedDepartureNumber)
print('..............')
print('All predicted transitions:')
for var in referenceResult:
    print(var)
print('..............')
print('Expected duration for',startStationID,'to',endStationID,':',predictedDuration,'s')
print('..............')
print('Time used:',time2-time1,'s')
