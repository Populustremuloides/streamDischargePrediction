import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# read in the data file
dataFile = "non_ref_burn_prediction_output.csv"

dfDict = {}
dfDict["dataType"] = []
dfDict["ecoregion"] = []
dfDict["catchment"] = []
dfDict["time_index"] = []
dfDict["flow"] = []

with open(dataFile, "r+") as predFile:
    firstLine = predFile.readline()
    firstLine = firstLine.split(",")
#    print(len(firstLine))

    for line in predFile:
        lineList = line.split(",")

        headers = lineList[0:6]
        dataType = str(headers[1])
        catchment = headers[2]
        
        ecoregion = int(catchment[0:2])

#        print(ecoregion)


        data = lineList[6:]
        
        time_index = 0
        for datum in data:
            if datum is None or datum.isspace() or datum == "None" or datum == "None\n":
                flow = None
            else:
                flow = float(datum)

            dfDict["flow"].append(flow)
            dfDict["dataType"].append(dataType)
            dfDict["ecoregion"].append(ecoregion)
            dfDict["catchment"].append(catchment)
            dfDict["time_index"].append(time_index)

            time_index += 1

def dateplot(x, y, **kwargs):
    ax = plt.gca()
    data = kwargs.pop("data")
    data.plot(x=x,y=y, ax=ax, grid=False, **kwargs)

df = pd.DataFrame.from_dict(dfDict)

ax = sns.lineplot(x="time_index",y="flow",hue="ecoregion",style="dataType",data=df)
#ax = sns.FacetGrid(df, col="ecoregion", col_wrap=1)
#ax = ax.map_dataframe(dateplot, "time_index" ,"flow")
plt.show()
