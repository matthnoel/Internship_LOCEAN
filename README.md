# Intership_LOCEAN

In this folder, you will find the different programs for post-processing the meteorological data from the ERA5 reanalyses. The objective is to verify to what extent the atmospheric model of the reanalyses takes into account the presence of Antarctic coastal polynyas.

The programs are characterised in two ways: those with a set of tool functions (ToolsPolynyas.py and MapDrawing.py) and those with which these functions are applied to the data sets.

 - ToolsPolynyas.py contains all the functions for working directly on the data sets, adapting and calculating the parameters of the reanalyses.

 - MapDrawing.py allows you to draw all the maps necessary for the study.

The other programs are used directly for the post-processing of the ERA5 reanalysis data.

 - analyse_temperature_stations.py : comparisons of the measured temperatures and those from the ERA5 reanalyses.
 - analyse_vent_stations.py : comparisons of measured wind speeds and those from ERA5 reanalyses.
 - AnalyseSpectrale.py : determination of the spectrum of the reanalysis data.
 - Anomalies_CD_2010-2020.py : calculation of anomalies (comparison of states: open & closed polynyas) for ERA5 atmospheric parameters for the Cape Darnley polynya.
 - Anomalies_MB_2010-2020.py : idem for the Mackenzie Bay polynya.
 - Comparaison_Tolérance_Polynie.py : Comparisons of the maximum tolerance thresholds for detecting polynyas.
 - Correlation_Vent_Ocean.py : calculation of the correlation between characteristic winds and mean values over a region of the ocean off Cape Darnley.
 - Correlation_Vent_Land.py : calculation of the correlation between the characteristic winds and the mean values over a region inland near Cape Darnley.
 - occurrences_polynies_cape_darnley.py : Calculation of the opening occurrence rate of the Cape Darnley polynya.
 - RegressionCapeDarnley.py: Calculation and plotting of atmospheric parameters against the polynya surface.
 - AverageMaps.py: Plot of atmospheric parameter maps from ERA5 reanalyses averaged over the winters of 2010 to 2020.
