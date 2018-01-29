

def dataframe_batch_generator(df, batch_size):
    a = 0
    b = min(batch_size, len(df))

    while b < len(df):

        yield df.iloc[a:b]

        a = b
        b = min(a + batch_size, len(df))