# STA9890 Spring 2020 Project

## Dataset Introduction
For this project, we would like to use related factors to predict life expectancy. We extracted World Development Indicators of twenty four countries from [World Bank Databank](https://databank.worldbank.org/source/world-development-indicators/preview/on). World Development Indicators (WDI) is the primary World Bank collection of development indicators, compiled from officially recognized international sources.

The response variable is Life Expectancy at birth (Series Code **SP.DYN.LE00.IN**), and the predictorsare the rest 41 variables, including GDP growth, Inflation, CO2 emissions, Physicians per 1,000 people, andHospital beds per 1,000 people, etc.

## Dataset Pre-processing
This dataset has 480 observations (n=480) and 42 variables (p=42). The following code has been implemented to impute missing data-points with their mean.

![alt text](https://raw.githubusercontent.com/juliazhu1122/sta9890project/master/img/5.png)

We used the following code to standardize the numerical predictors:

![alt text](https://raw.githubusercontent.com/juliazhu1122/sta9890project/master/img/1.png)
