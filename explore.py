#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

def feature_graphs(x):
    for col in x:
        print(col)
        ax = sns.histplot(x=col, data=x, kde=True)
        ax.lines[0].set_color('crimson')
        plt.show()

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