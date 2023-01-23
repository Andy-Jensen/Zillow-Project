# Zillow-Project
# An analysis of a continuous target variable

# Project Description:
## The scenario:
As a newly hired data scientist to the Zillow data science team, I am asked to analyze 'Single Family Properties' to make a model that predicts property value. I am asked to look at features and identify drivers of property value as well as make recommendations and tell next steps.

# Project Goals:
* Construct a ML Regression model that predicts property value of "Single Family Properties"
* Find key drivers of property value
* Deliver a replicable report that details steps taken
* Make recommendations on what works or doesn't work when predicting property values

# Initial Hypothesis:
I have minimally explored this dataset and my initial hypothisis is that homes with a higher square footage will have a higher property value

# Project Plan:

* Acquire the data from the Codeup SQL server

* Prepare data
   * Drop columns
       * longitude
       * latitude
   * Get dummies
       * bed_count
       * bath_count
   * Scale data
       * finished_sq_ft

* Separate into train, validate, and test datasets
 
* Explore the train data in search of potential drivers of property value
   * Answer the following initial questions
       * Compare the property value of homes with 2 bathrooms to the overall property value mean of the dataset (single sample ttest)
       * Compare the property value of homes with 3 bedrooms to the overall property value mean of the dataset (single sample ttest)
       * Subset homes <=3 bedrooms and >3 bedrooms to compare their means (2 sample/independent ttest)
       * How strong is the correlation between finished square feet and property value (Pearsonr)
       
* Develop a model that has a low RMSE to predict property value
   * Construct different types of regression models
   * Evaluate models on train and validate data
   * Select the best model based on the lowest difference between the train and validate RMSE
   * Run the test data on the selected model and check the test RMSE
 
* Draw conclusions

# Data Dictionary:

| Feature | Definition |
|:--------|:-----------|
|longitude| The longitudinal coordinate of the property|
|latitude| The latitudinal coordinate of the property|
|bath_count| The number of bathrooms the home has|
|bed_count| The number of bedrooms the home has|
|finished_sq_ft| How many calculated finished square feet the home has|
|property_value| The tax value dollar count of the property|

# Steps to Reproduce
1. Clone this repo
2. Use the function from acquire.py and prepare.py modules to obtain the data from the Codeup SQL server using the programmed query
3. Run the explore and modeling notebook
4. Run final report notebook using the explore.py and modeling.py modules
