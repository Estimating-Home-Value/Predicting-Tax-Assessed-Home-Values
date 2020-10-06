# What Drives Tax Assessed Home Values
# Delete later

View the google slide presentation here: [**Drivers for Tax Evaluation**](address)

## Description

You are a junior data scientist on the Zillow data science team and recieve the following email in your inbox:

> We want to be able to predict the values of single unit properties that the tax district assesses using the property data from those whose last transaction was during the "hot months" (in terms of real estate demand) of May and June in 2017. We also need some additional information outside of the model.

> Zach lost the email that told us where these properties were located. Ugh, Zach :-/. Because property taxes are assessed at the county level, we would like to know what states and counties these are located in.

> We'd also like to know the distribution of tax rates for each county.

> The data should have the tax amounts and tax value of the home, so it shouldn't be too hard to calculate. Please include in your report to us the distribution of tax rates for each county so that we can see how much they vary within the properties in the county and the rates the bulk of the properties sit around.

> Note that this is separate from the model you will build, because if you use tax amount in your model, you would be using a future data point to predict a future data point, and that is cheating! In other words, for prediction purposes, we won't know tax amount until we know tax value.

> -- The Zillow Data Science Team

You wonder how you can recieve an email from the entire data science team at once, but figure it's best to get started on the project.

## Goals
1. Identify significant drivers of tax assessment value.  
2. Describe and visualize the distribution of tax rates for each county. 
3. Produce a model that has a smaller root mean squared error than a model with no features based on the mean tax value. 
4. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. 
5. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

### Specification
#### Audience
Your customer is the zillow data science team. state your goals as if you were delivering this to zillow. They have asked for something from you and you are basically communicating in a more concise way, and very clearly, the goals as you understand them and as you have taken and acted upon through your research.

#### Deliverables
Remember that you are communicating to the Zillow team, not to your instructors. So, what does the team expect to receive from you?

1. A report (in the form of a presentation, both verbal and through a slides)

The report/presentation slides should summarize your findings about the drivers of the single unit property values. This will come from the analysis you do during the exploration phase of the pipeline. In the report, you should have visualizations that support your main points.

The presentation should be no longer than 5 minutes, and consist of 3-5 slides.

2. A github repository containing your work.

This repository should consist of at least 1 Jupyter Notebook that walks through the pipeline, but you may wish to split your work among 2 notebooks, one for exploration and one for modeling. In exploration, you should perform your analysis including the use of at least one statistical test along with visualizations documenting hypotheses and takeaways. In modeling, you should establish a baseline that you attempt to beat with various algorithms and/or hyperparameters. Evaluate your model by computing the metrics and comparing.

Make sure your notebooks answer all the questions posed in the email from the Zillow data science team.

The repository should also contain the .py files necessary to reproduce your work, and your work must be reproducible by someone with their own env.py file.

As with every project you do, you should have an excellent README.md file documenting your project planning with instructions on how someone could clone and reproduce your project on their own machine. Include at least your goals for the project, a data dictionary, and key findings and takeaways.

## Data Dictionary

| Column | Description |
| --- | ---|
| id | Unique id for each house |
| bathroomcnt | Number of Bathrooms; Includes halfbaths as 0.5 |
| bedroomcnt | Number of Bedrooms |
| calculatedbathnbr | Unknown |
| calculatedfinishedsquarefeet | Total square feet of home; doesn't include property square feet |
| fips | Federal Information Processing System codes used to identify unique geographical areas | 
| fullbathcnt | Number of full bathrooms |
| latitude | The latitude of the property
| longitude | The longitude of the property |
| yearbuilt | The year the house was built |
| taxvaluedollarcnt | The tax accessed value of the property in USD. |
| transactiondate | Four digit year, two digit month, two digit date | 

## Data Validation
The following considerations were taken with the data:
1. Initial SQL query produced 20,364 records that met the following requirements:
    * Transaction Date between 2017-05-01 and 2017-06-30
    * Property classified as one of the following Types:
        * Single Family Residential
        * Rural Residence
        * Mobile Home
        * Townhouse
        * Condominium
        * Row House
        * Bungalow
        * Manufactured, Modular, Prefabricated Homes
        * Patio Home
        * Inferred Single Family Residence
2. Records containing 0 bathrooms, 0 bedrooms, or null square footage were dropped
3. Duplicate records were dropped. These entries may represent "back-to-back" closings on the same day between three parties.

## Key Findings and Takeaways
1. All properties containing FIPS data are located in one of three Californian counties:
    * Los Angeles County
    * Orange County
    * Ventura County
2. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
3. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. 
4. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.

## How to Reproduce

### First clone this repo

### acquire.py 
* Must include `env.py` file in directory.
    * Contact [Codeup](https://codeup.com/contact/) to request access to the MySQL Server that the data is stored on.
    * `env.py` should include the following variables
        * `user` - should be your username
        * `password` - your password
        * `host` - the host address for the MySQL Server

### prep.py
* 

### model.py
* This file has n functions
    * 