#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def feature_graphs(train):
    '''
    first subplot of bathroom count
    '''
    plt.figure(figsize=(10,5))
    plt.subplot(221)
    sns.histplot(x=train['bath_count'], data=train)
    plt.title('Bathroom Count of Single Family Properties')
    plt.xlabel('Bathroom Count')
    '''
    second subplot of bedroom count
    '''
    plt.subplot(222)
    sns.histplot(x=train['bed_count'], data=train)
    plt.title('Bedroom Count of Single Family Properties')
    plt.xlabel('Bedhroom Count')
    '''
    third subplot of property value
    '''
    plt.subplot(223)
    ax = sns.histplot(x=train['property_value'], data=train, kde=True)
    ax.lines[0].set_color('crimson')
    plt.title('Property Values of Single Family Properties')
    plt.xlabel('Property Value')
    '''
    fourth subplot of finished square feet
    '''
    plt.subplot(224)
    ax = sns.histplot(x=train['finished_sq_ft'], data=train, kde=True)
    ax.lines[0].set_color('crimson')
    plt.title('Finished Square Feet of Single Family Properties')
    plt.xlabel('Finished Square Feet')
    '''
    adjusting the spacing between the subplots
    '''
    plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=1.5,
                    wspace=0.4,
                    hspace=0.4)

#first explore questions plots
def eq1_graphs(df):
    '''
    subset the data
    '''
    train_2b= df[df['bath_count']==2]
    '''
    set the figure size for the visuals
    '''
    plt.figure(figsize=(10,5))
    '''
    first plot-the subset data and property value
    '''
    plt.subplot(221)
    sns.histplot(x='property_value', data=train_2b)
    plt.title('2 Bathroom Homes')
    plt.xlabel('Property Value')
    plt.grid(True, alpha=0.3, linestyle='--')
    '''
    second plot-the property value of the overall data
    '''
    plt.subplot(222)
    sns.histplot(x='property_value', data=df)
    plt.title('All Homes')
    plt.xlabel('Property Value')
    plt.grid(True, alpha=0.3, linestyle='--')
    '''
    third plot-both of the previous two plots on top of eachother with the transparency turned down
    '''
    plt.subplot(223)
    plt.title('2 Bathroom Homes and All Homes')
    sns.histplot(x='property_value', data=train_2b, alpha=.75, color='green', label= '2 Bathrooms')
    sns.histplot(x='property_value', data=df, alpha=.25, label='All Homes')
    plt.xlabel('Property Value')
    '''
    add a vertical line to show the mean of the overall data's property value
    '''
    plt.axvline(x=(df['property_value'].mean()), color='red', label='Overall Mean')
    '''
    add a legend and gridlines to all plots
    '''
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.subplots_adjust(left=0.1,
                        bottom=-0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)


#first explore questions statistical test
def eq1_statistic(df):
    '''
    set the alpha value and subset the data
    '''
    alpha=.05
    train_2b= df[df['bath_count']==2]
    '''
    assigning variables and performing a one sample ttest
    ''' 
    sample= train_2b['property_value']
    overall_mean=df['property_value'].mean()
    t, p = stats.ttest_1samp(sample, overall_mean, alternative='less')
    '''
    output test results and wether to reject H_0 or not
    '''
    print(f't statistic= {round(t,2)}\np-value= {p}\nalpha= {alpha}')
    if p < alpha:
        print('Reject the null hyopthesis')
    else:
        print('Do not reject the null hypothesis')


#second explore questions plots
def eq2_graphs(df):
    '''
    subset the data
    '''
    train_3bed=df[df['bed_count']==3]
    '''
    set the figure size for the visuals
    '''
    plt.figure(figsize=(10,5))
    '''
    first plot-the subset data and property value
    '''
    plt.subplot(221)
    sns.histplot(x='property_value', data=train_3bed)
    plt.title('3 Bedroom Homes')
    plt.xlabel('Property Value')
    plt.grid(True, alpha=0.3, linestyle='--')
    '''
    second plot-the property value of the overall data
    '''
    plt.subplot(222)
    sns.histplot(x='property_value', data=df)
    plt.title('All Homes')
    plt.xlabel('Property Value')
    plt.grid(True, alpha=0.3, linestyle='--')
    '''
    third plot-both of the previous two plots on top of eachother with the transparency turned down
    '''
    plt.subplot(223)
    plt.title('3 Bedroom Homes and All Homes')
    sns.histplot(x='property_value', data=train_3bed, alpha=.75, color='green', label= '3 Bedroom')
    sns.histplot(x='property_value', data=df, alpha=.25, label='All Homes')
    plt.xlabel('Property Value')
    '''
    add a vertical line to show the mean of the overall data's property value
    '''
    plt.axvline(x=(df['property_value'].mean()), color='red', label='Overall Mean')
    '''
    add a legend and gridlines to all plots
    '''
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.subplots_adjust(left=0.1,
                        bottom=-0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

#function for second explore statistic
def eq2_statistic(df):
    '''
    set the alpha value and subset the data
    '''
    alpha=.05
    train_3bed=df[df['bed_count']==3]
    '''
    assigning variables and performing a one sample ttest
    ''' 
    sample= train_3bed['property_value']
    overall_mean=df['property_value'].mean()
    t, p = stats.ttest_1samp(sample, overall_mean, alternative='less')
    '''
    output test results and wether to reject H_0 or not
    '''
    print(f't statistic= {round(t,2)}\np-value= {p}\nalpha= {alpha}')
    if p < alpha:
        print('Reject the null hyopthesis')
    else:
        print('Do not reject the null hypothesis')

#Explore question 3 plots
def eq3_graphs(df):
    '''
    subset the data
    '''
    train_more3=df[df['bed_count']>3.5]
    train_less3=df[df['bed_count']<3.5]
    '''
    set the figure size for the visuals
    '''
    plt.figure(figsize=(10,5))
    '''
    first plot-the subset data and property value
    '''
    plt.subplot(221)
    sns.histplot(x='property_value', data=train_more3)
    plt.title('Homes With > 3.5 Bedrooms')
    plt.xlabel('Property Value')
    plt.grid(True, alpha=0.3, linestyle='--')
    '''
    second plot-the property value of the overall data
    '''
    plt.subplot(222)
    sns.histplot(x='property_value', data=train_less3)
    plt.title('Homes With < 3.5 Bedrooms')
    plt.xlabel('Property Value')
    plt.grid(True, alpha=0.3, linestyle='--')
    '''
    third plot-both of the previous two plots on top of eachother with the transparency turned down
    '''
    plt.subplot(223)
    plt.title('Homes With > 3.5 > Bedrooms')
    sns.histplot(x='property_value', data=train_more3, alpha=.75, color='green', label= 'Bedrooms > 3.5')
    sns.histplot(x='property_value', data=train_less3, alpha=.25, label='Bedrooms < 3.5')
    plt.xlabel('Property Value')
    '''
    add a vertical line to show the mean of the overall data's property value
    '''
    plt.axvline(x=(train_more3['property_value'].mean()), color='red', label='Bedrooms >3.5 Mean')
    plt.axvline(x=(train_less3['property_value'].mean()), color='yellow', label='Bedrooms <3.5 Mean')
    '''
    add a legend and gridlines to all plots
    '''
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.subplots_adjust(left=0.1,
                        bottom=-0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

#exploration question 3 statistic
def eq3_statistic(df):
    '''
    set the alpha value and subset the data
    '''
    train_more3=df[df['bed_count']>3.5]
    train_less3=df[df['bed_count']<3.5]
    alpha=.05
    '''
    checking for equal variance
    '''
    more3v= train_more3['property_value'].var()
    less3v= train_less3['property_value'].var()
    print(f'Variance of property values for homes with more than 3 bedrooms: {more3v}')
    print(f'Variance of property values for homes with less than 3 bedrooms: {less3v}')
    '''
    assigning variables and performing a one sample ttest
    ''' 
    more3m= train_more3['property_value']
    less3m=train_less3['property_value']
    t, p = stats.ttest_ind(less3m, more3m, alternative='less', equal_var=False)
    '''
    output test results and wether to reject H_0 or not
    '''
    print(f't statistic= {round(t,2)}\np-value= {p}\nalpha= {alpha}')
    if p < alpha:
        print('Reject the null hyopthesis')
    else:
        print('Do not reject the null hypothesis')

#explore question 4
def eq4_corr(df):
    '''
    set alpha
    '''
    alpha=0.05
    '''
    get the correlations and run a pearson r
    '''
    correlations=df.corr()
    corr, p = stats.pearsonr(df['property_value'], df['finished_sq_ft'])
    '''
    set output
    '''
    if p < alpha:
        print(f'The correlation between property value and finished square feet is {corr}')
    else:
        print('There is no correlation between property value and finished square feet.')
    return correlations