def main():
    import numpy as np
    heights=[1.5,1.7,1.6,1.2,1.2]
    heights_array=np.array(heights)
    print(heights_array)
    """
    Now there are various interesting calculations that we can do with numpy arrays
    Firstly if we do 
    
    npArray + npArray the result is a element wise addition of all the elements of the array
    Similarly we can do a divide between 2 numpy arrays and that will also result in an element wise division of the each array
    One important thing to note is that numpy arrays cannot store data of different types. For example if we have an int  a float and a string all will be converted to a string.
    
    Accessing elements of a numpy array: array[<index>]
    
    Another interesting feature:
    Lets say we want an list of all the elements in the numpy array that is greater than a specific number. In that case we can simply use the greater than sign in the following way:
    
    numpyArray>x
    The result is an array of booleans that says whether a particular element is greater than x
    """
    val=heights_array[heights_array>1.5]
    # the above line returns a array of values that are greater than 1.5 from heights_array
    print (val)
    
    """
    2D Numpy Array
    2d numpy arrays are made in a similar way as normal 2d lists in python
    """
    np_2d=np.array([[1.7,1.5,1.2,1],[1.2,1.9,2.0,2.1]])
    # so above is a 2d numpy array
    print(np_2d)
    print(np_2d.shape)
    # shape is an attribute of a numpy array that can give us the dimensions of numpy array
    print (np_2d[0])
    # the above line prints the first row
    print(np_2d[0][1])
    # as expected prints the first row , 2nd col 
    # also
    print(np_2d[0,2])
    # similarly
    print(np_2d[:,1:3])# remember in 1:3, 3 is not included, it just means 1 and 2 th element of both rows
    
    """
    Data Analysis:
    Mean: np.mean(<array>)
    Median: np.median(<array>)
    Correlation Coefficient: np.corrcoef(<array>,<array>)
    Standard deviation: np.std(<array>)
    Other functions are also there like sum and sort, random number generation functions are also there, check documentation for more insight
    """
    """
    Data Visulation:
    Library: matplotlib
    inside the matplotlib package there is pyplot sob package and this gives you all basic pyplot functionality
    
    """
    import matplotlib.pyplot as plt
    year=[2009,2010,2011,2012,2013]
    population=[1.7,1.8,1.4,1.4,1.8]
    plt.plot(year,population)
    # plot methods plots the data in the form of a line chart, the first list corresponds to x axis
    plt.show()
    # plot actually shows the plot to us
    plt.scatter(year,population)
    # the above code generates a scatter plot instead of a line plot 
    plt.show()
    """
    Data Analysis: concept of Histograms in pyplot
    There is a concept of bins in histograms in pyplot. Firstly seeing the documentation of the method hist we see  that the first 2 arguements are x and bins, x is the list of values to be considered, bins is the number of partitions the list x is divided into, python will automatically figure out how many values are there in each one of those bins
    
    In plotting different values it is important to note that the dimenstion of both x and y must be the same otherwise a ValueError is raised
    """
    x=[0,0.6,0.5,1.5,1.7,1.7,1.2,1.9,1.8,2.5,2.4]
    plt.hist(x,bins=4)
    plt.show()
    """
    Customizing your graphs
    """
    y=[1.5,2.3,1.3,1.8,1.9,2.5,2.5,1.7,2.9,2.5,1.2]
    # the plot method is only for plotting a line graph
    # plt.plot(x,y)
    plt.fill_between(x,y,0,color='green')
    # the above method is used for filling under the line graph, the 3rd arguement is for filling uptill 0
    plt.xlabel('x- axis')
    plt.ylabel('y- axis')
    plt.title('Sample graph')
    # we can change ticks on the y axis, think of ticks simply as the number of steps, the second arguement is the repersentation of # the ticks
    plt.yticks([0,0.25,0.5,0.75,1,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0])
    plt.show()
    
    """
    Pandas Package:
    High level data manipulation tool used by data scientists. We can store data in pandas in data frames, data frames are tables
    Generally data in data frames are not manually entered, rather they are entered from files especially csv files
    """
    import pandas as pd
    # for reading from a csv file into a data frame we can simply
    # x=pd.read_csv(<path to csv>, index_col=<column number in which the row indices are stored>)
    # important thing to note in the read_csv function is that the path to csv file should be the absolute path
    # Column access: <name of data frame object>.<name of column as a property of the object itself> or <name of data frame object>[<name of the column>]
    # adding a column: <name of the data frame object>[<name of the column>]=[<list of values for each row>]
    # Example datFrame["col1"]=[row0Val,row1Val,row2Val]
    # Pandas is based on numpy and so we can carry out operations like divide one list(or really a numpy array) by annother
    # Accessing a row: <name of the data frame object>.loc[<row name that is the first column>]
    # Accesssing elements: <name of the data frame object>.loc[<row name>,<column name>]