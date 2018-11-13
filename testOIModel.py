import csv
import datetime
import matplotlib.pyplot as plt
import OIModel as om
import time

time1 = time.time()
# type in file loc
transitionDataLoc = 'C:/Users/Tony Xu/OneDrive/KDD 2/dataOIModel.csv' # transition file
stationStatusDataLoc = 'C:/Users/Tony Xu/OneDrive/KDD 2/stationStatus.csv' # station status file
weatherDataLoc = 'C:/Users/Tony Xu/OneDrive/KDD 2/weather.csv' # weather info file

# Initialize function, no need to modify
hisInputData, transitionMatrixDuration, transitionMatrixDetination, weatherMatrix = om.read_inData(transitionDataLoc,stationStatusDataLoc,weatherDataLoc)

# Inputs time stamp : e.g. 1373964540
# Inputs station ID : e.g.212
# return: expected departure(rent) number of this station in a period(hour)
expectedDepartureNumber = om.get_expectDepartureNumber(1373964540, 212,hisInputData,weatherMatrix)

# Inputs time stamp : e.g. 1373964540
# Inputs station ID : e.g.212
# return: expected destination and departure time
predictedDestination,expectedDepartureTime = om.get_predictedDestination(1373964540, 212,transitionMatrixDuration, transitionMatrixDetination)

# Inputs time stamp : e.g. 1373964540
# Inputs start station ID : e.g.212
# Inputs end station ID : e.g.404
# return: expected time used
predictedDuration = om.get_predictedDuration(1373964540, 212, predictedDestination,transitionMatrixDuration, transitionMatrixDetination)
time2 = time.time()

print('Time used:',time2-time1,'s')
print(expectedDepartureNumber)
print(predictedDestination,expectedDepartureTime)
print(predictedDuration)