# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 13:26:57 2017

@author: riflerRick
"""

"""
Now that we know a little bit of numpy, matplotlib and pandas lets try and simulate a few values here first and then work on them
"""
"""
This first function is going to simulate data points for us. First thing to note here is that just by looking at how points are being generated and does not actually give us idea of what we are finally gonna achieve, our final goal is to make a dataframe that will contain x and y as the 2 attributes for each entry and a label Z. Now based on whether the label is 0 or 1 we are making 2 different distributions so that we can get a fair amount of dissimilarity between the points in the graph, they should not be too inter-mingled for the purposes of the simulation.
"""
def sim_log_data(meanX1,meanY1,totalPointsN1,sd1,meanX2,meanY2,totalPointsN2,sd2):
    import pandas as pd
    import numpy.random as nr
    wx1=nr.normal(loc=meanX1,scale=sd1,size=totalPointsN1)
    wy1=nr.normal(loc=meanY1,scale=sd1,size=totalPointsN1)
    # the above code simulates points (x1,y1) for the labels z1 which we can decide to be either 1 or 0
    # consult documentation for nr.normal but for a gist the parameters are (mean, sd, total points to be generated). Returns 
    # numpy ndarray which will store points generated out of a normal distribution
    z1=[0]*totalPointsN1 
    # z1 is essentially a list of 0s totalPointsN1 times
    wx2=nr.normal(loc=meanX2,scale=sd2,size=totalPointsN2)
    wy2=nr.normal(loc=meanY2,scale=sd2,size=totalPointsN2)
    z2=[1]*totalPointsN2

    # considering z1 to be all 0s and z2 to be all 1s is completely arbitrary

    # now we build a dataframe of all these points generated
    df1=pd.DataFrame({'x':wx1,'y':wy1,'z':z1})
    df2=pd.DataFrame({'x':wx2,'y':wy2,'z':z2})

    # so the DataFrame module accepts a dictionary with the keys as the column headers and values as lists containing records or # rows

    return pd.concat([df1,df2],axis=0,ignore_index=True)

    # pd.concat simply concatenates 2 dataframes here, axis =0 means concatenation takes place one below the other

def plot_class(df):
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(8,8))
    # (8,8) is a (w,h) tuple which says that the figure size will be 8 by 8 inches, retuns a Figure instance
    fig.clf()
    # clf clears the current figure
    ax=fig.gca()
    # create axes
    df[df.z==1].plot(kind='scatter',x='x',y='y',ax=ax,alpha=1.0,color='Red',marker='x',s=40)
    df[df.z==0].plot(kind='scatter',x='x',y='y',ax=ax,alpha=1.0,color='DarkBlue',marker='o',s=40)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Classes vs X and Y')
    return 'Done'
"""
On plotting the garph one thing that we can notice is that the random nature 
of the plot surely means that there is no striaght forward classification of 
the data, So we will use the logsitic regression model in order to train the 
machine so that it classify points accordingly
"""

def logistic_mod(df,logProb=1.0):
    from sklearn import linear_model
    """
    The first step is essentially preparing the data 
    """
    # df.shape actually gives the dimensions of the dataframe in the form 
    # of a tuple, extracting values from a tuple is using square brac notation
    nrow=df.shape[0]
    # nrow is the number of rows in the dataframe
    X=df[['x','y']].as_matrix().reshape(nrow,2)
    # here X represents all the attributes
    # reshape kinda forces the array to be of shape nrow,2
    Y=df.z.as_matrix().ravel()
    # ravel returs a contiguous flattened array
    """
    Compute the logistic regression model
    """
    lg=linear_model.LogisticRegression()
    # so we make the logistic regression model and name it lg
    logr=lg.fit(X,Y)
    # then we try to fit a curve y=f(x), here Y is the labels
    temp=logr.predict_log_proba(X)
    # temp is a numpy array, predict_log_proba actually calulates the points
    # based on the decision boundary made
    df['predicted']=[1 if (logProb >p[1]/p[0]) else 0 for p in temp]
    # we can simply append to a dataframe and therefore add a column using 
    # 'predicted' as the column name 
    return df
def testFunction():
    return 'hurray'
           
def main():
    df=sim_log_data(-1,-1,10,1,1,1,10,1)
    plot_class(df)

if __name__=="__main__":
    main()