import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    df_unit_size = pd.read_csv("../Data/Unit_Cell_Sizes.csv") # data containing size of 
    df_endpoint = pd.read_csv("../Data/MacrophagesforML.csv")
    # if the average bacteria count in 800*800 is 0.4, then how many bacteria will be in 560*560
    x = (560/800)**2
    print(x)
    # join the two dataframes on the basis of Feature ID and FeatID
    merged_df = pd.merge(df_endpoint,df_unit_size, left_on="FeatID", right_on="Feature ID", how="left")
    merged_df = merged_df.drop("Feature ID", axis=1)
    print(merged_df.head(10))
    
    # Average equal to Average for unit cell sizes 10 and 20, and (560/800)^2 * Average for unit cell size 28
    df = merged_df[['FeatID', 'Average']]
    df['Average'] = np.where(merged_df['Feature Unit Cell Size'] == 28,  merged_df['Average'] * x, merged_df['Average'])
    print(df.head(10))

    df.to_csv("../Data/MacrophagesByImageSize.csv", index=False)
    


if __name__ == "__main__":
    main()
    