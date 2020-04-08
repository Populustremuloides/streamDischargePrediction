import sys

diffFile = sys.argv[1]
outputFile = sys.argv[2]

oneBeforeFire = 0
yearOfFire = 365
yearOneAfter = 730
yearTwoAfter = 1095
yearThreeAfter = 1460
yearFourAfter = 1825
yearFiveAfter = 2190
yearSixAfter = 2555
yearSevenAfter = 2920
yearEightAfter = 3285
yearNineAfter = 3650
yearTenAfter = 4015



print(diffFile)
with open(diffFile, "r+") as dFile:
    firstLine = dFile.readline()

    for line in dFile:
        line = line.split(",")






