# chart_maker.py
#
# This Python script contains helper functions for grouping and charting.
#

# Import libraries.
#
import pandas as pd
import altair as alt

# Declare global variables.
#
figure_number = 1
table_number  = 1
    
# create_dataframe_group_by_columns_and_agg
#
def create_dataframe_group_by_columns_and_agg(df, group_args, agg_column, agg_arg):
    """
    
    Given a dataframe, grouping columns, a column on which to aggregate data
    and the type of aggregate function to invoke, this function creates a sorted 
    data grouping.
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        The dataframe on which to group data
    group_args : list
        The columns to group
    agg_colummn : str
        The column on which to invoke the aggregate function
    agg_arg : str
        The aggregate function to invoke
        
    Returns
    -------
    pandas.core.frame.DataFrame 
        The grouped dataframe

    Raises
    ------
    TypeError
        If the input argument df is not of type DataFrame

    Exception
        If the input argument agg_column does not exist in the dataframe or the
        aggregate function is not valid

    Examples
    --------
    >>> create_dataframe_group_by_columns_and_agg(movie_merge_df, ['genre'], 'movie_title', 'count')
    """
    
    global table_number
    
    # Ensure that the dataframe parameter is indeed a dataframe.
    #
    if type(df) is not pd.core.frame.DataFrame:
        raise TypeError("The first parameter needs to be a dataframe")

    # Verify that the x_axis exists in the dataframe.
    #
    if agg_column not in df.columns.tolist():
        raise Exception("Column" + agg_column + "does not exist in the dataframe.")
    
    # Ensure that the specified aggregate function is valid.
    #
    agg_arg_list = ['count', 'sum', 'mean', 'max', 'min']
    
    if agg_arg not in agg_arg_list:
        
        raise Exception("Aggregate function", agg_arg,"is not valid.")
    
    print ("Table", table_number, ":", agg_arg, 'on', agg_column, 'for the group', group_args)
    
    table_number = table_number + 1
    
    # Create a grouping based on the group arguments, aggregate column and the
    # aggregate arguments (e.g., mean, count).
    #
    p_df = (df.groupby(by=group_args).agg({agg_column : agg_arg}).
            sort_values(by=agg_column, ascending=False)
            )

    return p_df

    
# plot_barchart
#
def plot_barchart(df, graph_title, x_axis, x_title, y_axis, y_title, agg_func):
    """
    
    Given a dataframe, title, X axis, X axis title, Y axis, Y axis title and an
    aggregate function, this function creates a bar chart.
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        The dataframe to plot
    graph_title : str
        The bar chart title
    x_axis : str
        The bar chart X axis column
    x_title : str
        The bar chart X axis title
    y_axis : str
        The bar chart Y axis column
    y_title : str
        The bar chart Y axis title
    agg_func : str
        The aggregate function to invoke
        
    Returns
    -------
    altair.vegalite.v4.api.Chart 
        the plotted histogram

    Raises
    ------
    TypeError
        If the input argument df is not of type DataFrame

    Exception
        If the input argument x_axis or y_axis does not exist in the dataframe

    Examples
    --------
    >>> my_graph = plot_barchart(movie_merge_df, 'Number of Movies by Genre', 
                                'genre', 'Genre', 'movie_title', 'Movies', 'count')

    altair.vegalite.v4.api.Chart 
    """

    global figure_number
    
    # Ensure that the dataframe parameter is indeed a dataframe.
    #
    if type(df) is not pd.core.frame.DataFrame:
        raise TypeError("The first parameter needs to be a dataframe")

    # Verify that the x_axis exists in the dataframe.
    #
    if x_axis not in df.columns.tolist():
        raise Exception("Column" + x_axis + "does not exist in the dataframe.")

    # Verify that the y_axis exists in the dataframe.
    #
    if y_axis not in df.columns.tolist():
        raise Exception("Column" + y_axis + "does not exist in the dataframe.")
    
    # Invoke the applicable aggregate function based on the specified argument.
    #
    if agg_func == 'count':
        
        my_group = df.groupby(by=x_axis).count().loc[:, y_axis]

    elif agg_func == 'mean':
    
        my_group = df.groupby(by=x_axis).mean().loc[:, y_axis]
        
    elif agg_func == 'sum':
        
        my_group = df.groupby(by=x_axis).sum().loc[:, y_axis]

    elif agg_func == 'max':
        
        my_group = df.groupby(by=x_axis).max().loc[:, y_axis]
        
    elif agg_func == 'min':
        
        my_group = df.groupby(by=x_axis).min().loc[:, y_axis]

    else:
        
        raise Exception("Aggregate function", agg_func, "is not valid.")
    
    my_group = my_group.reset_index()

    # Plot the bar graph.
    #
    my_graph = alt.Chart(my_group, width=500, height=300).mark_bar().encode(
                         x=alt.X(x_axis + ':N', sort='y', title=x_title), 
                         y=alt.Y(y_axis + ':Q', title=y_title)
                        ).properties(title="Figure " + str(figure_number) + ": " + graph_title)

    figure_number = figure_number + 1
    
    return my_graph
    

# plot_scatter_chart
#
def plot_scatter_chart(df, x_axis, y_axis, graph_title="Untitled"):
    """
    
    Given a dataframe, X axis, Y axis and graph title, this function creates
    a scatter chart.
    
    Parameters
    ----------
    df : pandas.core.frame.DataFrame
        The dataframe to plot
    x_axis : str
        The scatter plot X axis column
    y_axis : str
        The scatter plot Y axis column
    graph_title : str
        The scatter plot title
        
    Returns
    -------
    altair.vegalite.v4.api.Chart 
        the plotted scatter plot
        
    Raises
    ------
    TypeError
        If the input argument df is not of type DataFrame

    Exception
        If the input argument x_axis or y_axis does not exist in the dataframe

    Examples
    --------
    >>> my_graph = plot_scatter_chart(movie_merge_df, 'Year', 'total_gross_dollars', 
                                      'Total Gross Revenue by Year')

    altair.vegalite.v4.api.Chart 
    """
        
    global figure_number

    # Ensure that the dataframe parameter is indeed a dataframe.
    #
    if type(df) is not pd.core.frame.DataFrame:
        raise TypeError("The first parameter needs to be a dataframe")

    # Verify that the x_axis exists in the dataframe.
    #
    if x_axis not in df.columns.tolist():
        raise Exception("Column" + x_axis + "does not exist in the dataframe.")

    # Verify that the y_axis exists in the dataframe.
    #
    if y_axis not in df.columns.tolist():
        raise Exception("Column" + y_axis + "does not exist in the dataframe.")

    # Create the scatter plot.
    #
    my_scatter = alt.Chart(df).mark_circle(color="slateblue", size=12, opacity=.3).encode(
                            x=alt.X(x_axis, bin=alt.Bin(maxbins=25)),
                            y=alt.Y(y_axis)).properties(
                            title="Figure " + str(figure_number) + ": " + graph_title)
    
    figure_number = figure_number + 1

    return my_scatter


