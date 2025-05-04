import xgboost
import numpy as np
import pandas as pd
model = xgboost.XGBClassifier()
model.load_model('fixation_model.json')

def build_Xy(df, window = 50):

    GAZE_X='x'
    GAZE_Y='y'
    FIX_X="fix_x"
    FIX_Y="fix_y"
    FIX="FIX"
    TIME="t"

    X_columns = [GAZE_X, GAZE_Y]
    Y_column = FIX
    
    X = df[X_columns].ewm(span = window).mean()
    for col in [GAZE_X, GAZE_Y]:
        X[col] = X[col].fillna(-1).astype(float)

        name = f"{col}_var"
        df[name] = df[col] - df[col].rolling(window,center=True).mean()
        X_columns.append(name)
        
        for i in range (-1*window,window):
            name = f"{col}_{i}"
            df[name] = df[col].diff().shift(i)
            X_columns.append(name)
            df = df.copy()
    
    name = f"var_diff"
    df[name] = np.abs( df[f'{GAZE_X}_var'] - df[f'{GAZE_Y}_var'] )
    X_columns.append(name)
    
    df[Y_column] = 1
    
    return df, X_columns, Y_column


def extract_fixations(x,y,t):
    xy =  np.column_stack((x, y))
    df = pd.DataFrame(xy, columns=['x', 'y'] )
    
    df, X_columns, _ = build_Xy(df, window = 50)

    print("Feature engineering - colummns:", X_columns)
    y_pred = model.predict(df[X_columns])

    return y_pred