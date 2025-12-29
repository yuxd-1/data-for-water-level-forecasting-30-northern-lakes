# data-for-water-level-forecasting-30-northern-lakes
This repository deposits data and code relevant to the paper "Evaluating Long-Term Forecasting Performance Decay and Seasonal Effects in Deep Learning-Based Water Level Forecasting Models Across Multiple Northern Lakes" 
## Meteological and water level observations (folder METEO and WL)
The water level observations, from the Finnish Environment Institute (Syke) Open data, are used as part of input (historical water level), output label, and water dynamics analysis.
<br>
<br>
Meteorological input features are extracted from the netcdf files from open data services of the Finnish IT center for science, https://paituli.csc.fi and Finnish Meteological Institute, https://www.ilmastokatsaus.fi/2022/07/06/daily-gridded-evapotranspiration-data-for-finland-for-19812020/
## Performance metrics of the forecasting models (folder gru_metrics and lstm_metrics)
NSE, RE, RMSE scores for each lake from lead time = 1 to lead time = 15 and across 12 months
### The repository also stores code and requirements.txt for hyperparameter tuning of the two regional models

