# test_chart_maker_test.py
#
# This script tests the primary chart_maker function.
#

# Import libraries.
#
import pandas as pd
from chart_maker import create_dataframe_group_by_columns_and_agg


# test_create_dataframe_group_by_columns_and_agg
#
def test_create_dataframe_group_by_columns_and_agg():
    """
    
    This function tests the create_dataframe_group_by_columns_and_agg
    function in chart_maker.py.
    
    Parameters
    ----------
    N/A
        
    Returns
    -------
    True - if all tests pass, the function will return to the calling entity
    
    Raises
    ------
    N/A
    
    Examples
    --------
    >>> test_create_dataframe_group_by_columns_and_agg()
    """    
    
    # Create the test dataframe.
    #
    df_info = {'movie_title': ['Jaws', 'Star Wars', 'Raiders of the Lost Ark'], 
            'year': [1975, 1977, 1981],
            'genre': ['Thriller','Adventure','Adventure'], 
            'MPAA_rating': ['PG-13', 'PG', 'PG']}
    df = pd.DataFrame.from_dict(df_info)

    # Test the count function (dataframe shape is correct).
    #
    assert create_dataframe_group_by_columns_and_agg(df, ['genre'], 'movie_title', 'count').shape == (2, 1), "There should be 2 rows of 1 column."
    
    # Test the min function for the year for the Adventure genre.
    #
    assert create_dataframe_group_by_columns_and_agg(df, ['genre'], 'year', 'min').iloc[0,0] == 1977, "The earliest date for the Adventure genre is 1977."

    # Test the max function for the year for the Adventure genre.
    #
    assert create_dataframe_group_by_columns_and_agg(df, ['genre'], 'year', 'max').iloc[0,0] == 1981, "The latest date for the Adventure genre is 1981."
    
    # Test the sum function for the year for the Adventure genre.
    #
    assert create_dataframe_group_by_columns_and_agg(df, ['genre'], 'year', 'sum').iloc[0,0] == 3958, "The sum of the years for the Adventure genre is 3958."

    return

