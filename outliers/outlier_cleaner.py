#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []

    ### your code goes here

    import pandas as pd
    df1 = pd.DataFrame(predictions, columns=['predictions',])
    df2 = pd.DataFrame(ages, columns=['age',])
    df3 = pd.DataFrame(net_worths, columns=['net_worth',])

    df = pd.concat([df1, df2, df3], axis=1)
    df['error'] = (df['predictions'] - df['net_worth']).abs()
    df = df.sort_values('error').head(int(len(df) * 0.9))

    cleaned_data = list(df.apply(lambda x: (x['age'], x['net_worth'], x['error']), axis=1))

    return cleaned_data
