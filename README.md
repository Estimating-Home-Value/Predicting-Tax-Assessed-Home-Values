# What Drives Tax Assessed Home Values

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
4. Identify strengths and weaknesses of final model.
5. Detail recommendations for future improvement.

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
| id | Autoincremented unique index id for each property |
| bathroomcnt | Number of Bathrooms; Includes halfbaths as 0.5 |
| bedroomcnt | Number of Bedrooms |
| calculatedbathnbr | Precise meaning unknown, but appears to be redundant with bathroomcnt and bedroomcnt |
| calculatedfinishedsquarefeet | Total square feet of home; doesn't include property square feet |
| finishedsquarefeet12| Unknown, but appears to be redundant with calculatedfinishedsquarefeet | 
| fips | Federal Information Processing System codes used to identify unique geographical areas | 
| fullbathcnt | Number of full bathrooms |
| latitude | The latitude of the property
| longitude | The longitude of the property |
| lotsizesquarefeet| The size of the total property lot |
| propertycountylandusecode | Unknown, but represents categorical government code |
| propertylandusetypeid |  Categorical variable describing the general type of property |
| rawcensustractandblock | Government id for each property linked to geographic location |
| regionidcity | Categorical variable identifying geographic location |
| regionidcounty | Categorical variable identifying geographic location |
| roomcnt | Number of rooms |
| yearbuilt | The year the house was built |
| structuretaxvaluedollarcnt | The tax assessed value of only the property structure in USD | 
| assessmentyear | Year that the tax value was assessed |
| landtaxvaluedollarcnt | The tax assessed value of only the land lot for the property |
| taxamount | The amount paid in taxes by the landowner in USD |
| taxvaluedollarcnt | The tax accessed value of the property in USD |
| censustractandblock | Redundant with rarcensustractandblock |
| logerror | Unknown |
| transactiondate | Four digit year, two digit month, two digit date | 
| taxrate | Rounded derived value by dividing the taxamount by the taxvaluedollarcnt and multiplying by 100 |
| County | County the property is located in | 

Additional columns were present in the zillow database but had greater than 20% null values and were dropped during initial consideration. 

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
2. Los Angeles County has both the highest mean tax rate and the highest range of tax rates. 
3. Los Angeles County has the lowest tax assessed property values. 
4. The strongest features identified by recursive feature elimination using linear regression are:
    * bedroomcnt
    * calculatedfinishedsquarefeet
    * latitude
    * fips (split into dummy variables for Los Angeles County, Orange County, and Ventura County)
5. A linear regression model with 3rd degree polynomial features performed the best out of the algorithms tested. When compared to a baseline model of no features based on the mean tax assessed property value, the model had 56% of the root mean squared error the baseline model had. 
6. For future improvements, additional exploration of features could be performed. Binning categorical features with a large number of unique values into relevant supercategories could prove useful. Additional changes to hyperparameters may also improve performance.
7. The model performs best on midrange homes, but has the weakest performance on low value properties. This is likely due to the influence of high value outliers. The model could be improved on its performance on lower value properties (which also make up a larger proportion of the overall property distribution relative to the higher value properties) by removing outliers or using a scaling method that is more robust to outliers.
8. Some limitations of the model are based on the use of tax valuation itself. While tax value is updated on a yearly basis in the area, California Proposition 13 limits increases to the valuation of a property to 2% per year. If actual home values based on sales and purchases are trending higher at faster than 2% per year, the data will be unable to accuractely capture the increased property value.

## How to Reproduce

### First clone this repo

### acquire.py 
* Must include `env.py` file in directory.
    * Contact [Codeup](https://codeup.com/contact/) to request access to the MySQL Server that the data is stored on.
    * `env.py` should include the following variables
        * `user` - should be your username
        * `password` - your password
        * `host` - the host address for the MySQL Server

### prepare.py
### features.py
* The functions in prepare.py and features.py can be imported to another file. Each function is specific to the task developed during the data science pipeline of this project and may need to be altered to suit different purposes. 
### model.ipynb
* There are several specific functions embedded in the model.ipynb. They will need to be copied to a .py file in order to be used elsewhere.