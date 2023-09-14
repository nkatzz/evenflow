import pandas as pd
import numpy as np

def get_directions(df: DataFrame):
    """Generates (x,y,z)-directions plus a single direction value from the quaternions"""

     
    zipped = zip(df.ox, df.oy, df.oz, df.ow)
    
    # A quaternion is normalized if its magnitude is 1.0 
    magnitudes = [np.sqrt(x**2 + y**2 + z**2 + w**2) for x, y, z, w in zipped]
    is_normalized = all(x == 1.0 for x in magnitudes)
    
    if not is_normalized:
       # To normalize a quaternion we need to divide each of its coords with its magnitude
       for column in ['ox', 'oy', 'oz', 'ow']:
           df[column] = df[column].div(magnitudes)
    else:
       	
    return normalized
    

df = pd.read_csv('/media/nkatz/storage/EVENFLOW-DATA/DFKI/new-3-8-2023/DemoDataset_1Robot.csv')
zipped = zip(df.ox, df.oy, df.oz, df.ow)
df['magnitude'] = [m.sqrt(x**2 + y**2 + z**2 + w**2) for x, y, z, w in zipped]


