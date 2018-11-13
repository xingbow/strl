import csv
import datetime
import matplotlib.pyplot as plt
import OIModel as om
import time

time1 = time.time()
hisInputData, transitionMatrixDuration, transitionMatrixDetination, weatherMatrix = om.read_inData()




# rent demands
expectedDepartureNumber,expectedDepartureTime = om.get_expectDepartureNumber(1373964540, 212,hisInputData,weatherMatrix)

# likly return station
predictedDestination = om.get_predictedDestination(1373964540, 212,transitionMatrixDuration, transitionMatrixDetination)

# duration
predictedDuration = om.get_predictedDuration(1373964540, 212, predictedDestination,transitionMatrixDuration, transitionMatrixDetination)
time2 = time.time()

print('Time used:',time2-time1,'s')
print(expectedDepartureNumber,expectedDepartureTime)
print(predictedDestination)
print(predictedDuration)


# 
# b -> a (full)
# b -> c 
#