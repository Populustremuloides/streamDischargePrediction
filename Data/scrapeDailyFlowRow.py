import requests
import datetime
import csv
# "https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites=06090800&period=P72000D"

outFile = "02_output_old_included.csv"
startYear = 1984 # must be a leap year!

def dateToIndex(date):
    year, month, day = date.split("-")
    year = int(year)
    month = int(month)
    day = int(day)

    index = 0

    # add the years *************************************************
    numYears = year - startYear
    for yearSinceStart in range(numYears):
        if yearSinceStart == 0:
            if startYear % 4 == 0: # if the start year was a leap year
                daysInYear = 366
            else:
                daysInYear = 365
        elif (yearSinceStart % 4) == 0:
            daysInYear = 366
        else:
            daysInYear = 365

        index = index + daysInYear

    # add the months ******************************************
    if year % 4 == 0: # if it is a leap year
        monthToDays = {1:31, 2:29, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
    else:
        monthToDays = {1:31, 2:28, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}

    monthVal = int(month)

    if monthVal > 1:
        for month in range(monthVal - 1): # don't include the current month because it isn't over yet!
            month = month + 1 # start in January, not month 0
            index = index + monthToDays[month]

    # add the days *******************************************
    index = index + day

    return index

# goodLink = "https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites=06090800&period=P50D"

link1 = "https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites="
link2 = "&period=P"

today = datetime.datetime.today()
today = str(today)
date, time = today.split(" ")
currentIndex = dateToIndex(date) - 1
days = currentIndex # Gives us records back to Jan 1, 1980
link3 = "D"

catchments = []
with open("all_catchments.csv") as catchmentFile:
    for line in catchmentFile:
        line = line.strip("\n")
        if len(line) == 7:
            line = "0" + line
        catchments.append(line)

links = []
for catchment in catchments:
    link = link1 + catchment + link2 + str(days) + link3
    links.append(link)

def getHeaderIndices(headers):
    j = 0
    for header in headers:
        if "site_no" in header:
            siteIdIndex = j
        elif "datetime" in header:
            dateIndex = j
        elif "_00060_" in header and "_cd" not in header:
            flowIndex = j
        elif "_00060_" in header and "_cd" in header:
            qualityIndex = j
        j = j + 1
    return siteIdIndex, dateIndex, flowIndex, qualityIndex

def getLineData(inputRow, siteIndex, dateIndex, flowIndex, qualityIndex):
    site = inputRow[siteIndex]
    date = inputRow[dateIndex]
    flow = inputRow[flowIndex]
    quality = inputRow[qualityIndex]
    return site, date, flow, quality


def initializeOutputFile(outFile):
    with open(outFile, "w+") as of:
        header = "catchment_id"
        for index in range(days):
            header = header + "," + str(index)
        of.write(header + "\n")

initializeOutputFile(outFile)

def linkDataToString(link, catchment):
    data = [catchment]

    # get the data from the current link
    f = requests.get(link)
    inputRows = f.text.split("\n")

    if len(inputRows) < 10: # if the link was a dud
        return data

    # scan to make sure it contains the right kind of data
    numLinesCopied = 0
    numGoodLines = 0
    for i in range(len(inputRows)):

        inputRow = inputRows[i]

        if not inputRow.startswith("#"): # only copy lines with actual data
            combinedLine = inputRow
            inputRow = inputRow.split("\t")

            # grab the indices of the various data points
            if numGoodLines == 0:

                if "_00060_" not in combinedLine: # if it doesn't contain the right kind of data
                    return data

                siteIndex, dateIndex, flowIndex, qualityIndex = getHeaderIndices(inputRow)
                numGoodLines += 1

            elif numGoodLines == 1:
                # skip the junk line
                numGoodLines += 1

            else:
                # get the data for that row
                if len(inputRow) > 1:

                    # extract the current information from here
                    site, lineDate, flow, quality = getLineData(inputRow, siteIndex, dateIndex, flowIndex, qualityIndex)
                    index = dateToIndex(lineDate)

                    if int(numLinesCopied) < index: # if there is a gap

                        while int(numLinesCopied) < index:
                            data.append(",") # append a new comma followed by blank data
                            numLinesCopied += 1

                        # now that we're caught up, write down the new data
                        if "A" in quality:
                            data.append("," + str(flow))
                        else:
                            data.append(",")

                        numLinesCopied += 1

                    elif int(numLinesCopied) == int(index): # if the two line up
                        if "A" in quality:
                            data.append("," + str(flow))
                        else:
                            data.append(",")
                        numLinesCopied += 1

                    else:
                        print("THERE WAS A MISTAKE")
                    numGoodLines += 1

    while len(data) <= days:
        data.append(",")
        numLinesCopied += 1

    return data

with open(outFile, "a+") as oFile:
    for j in range(len(links)):
        link = links[j]
        catchment = catchments[j]
        print(link)
        rowData = linkDataToString(link, catchment)
        rowData = rowData[0:days]
        print(len(rowData))
        if len(rowData) > 100:
            rowString = "".join(rowData)
            oFile.write(rowString + "\n")


