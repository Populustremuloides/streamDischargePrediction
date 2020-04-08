allCatchments = "all_catchments.csv"

def fixFile(inFileName, outFileName):
    outFile = open(outFileName, "w+")
    inFile = open(inFileName, "r+")
    for line in inFile:
        line = line.split(",")
        catchment = line[0]
        if len(catchment) == 7:
            catchment = "0" + catchment
        line = ",".join(line)
        line = line
        outFile.write(line)
    outFile.close()
    inFile.close()

fixFile(allCatchments, "all_catchments_corrected.csv")
fixFile("reference_watersheds.csv", "reference_watersheds_corrected.csv")
fixFile("normalized_zerod_catchment_characteristics.csv", "normalized_zerod_catchment_characteristics_corrected.csv")
fixFile("corrected_catchment_characteristics_corrected.csv", "corrected_catchment_characteristics_corrected.csv")
fixFile("catchments_for_Q_analysis.csv", "catchments_for_Q_analysis_corrected.csv")
