
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.rand(4,5), columns = list('abcde'))
df_list = [df, df]

np_array = np.array(list(map(lambda x: x.to_numpy(), df_list)))
#np.array([np.array(df), np.array(df)])

# to make sure the shape of np_array is correct
np_array = np_array.reshape((28, 28, 19590))
