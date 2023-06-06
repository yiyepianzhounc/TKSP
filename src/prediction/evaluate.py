import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score


def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = np.sqrt(mean_squared_error(y_test, y_predict))
    mape = (abs(y_predict - y_test) / y_test).mean().item()
    r_2 = r2_score(y_test, y_predict)
    return mae, rmse, mape, r_2
def diff_evaluation(y_test, y_predict):
    print(len(y_predict), len(y_test))
    y_test =[i-j for i,j in zip(y_test[1:],y_test[:len(y_test)-1])]
    y_predict=[i-j for i,j in zip(y_predict[1:],y_predict[:len(y_predict)-1])]
    print(len(y_predict),len(y_test))
    y_predict=pd.DataFrame(y_predict).diff().dropna()
    y_test=pd.DataFrame(y_test).diff().dropna()
    print(type(y_test))
    return evaluation(y_test,y_predict)

#mae, rmse, mape, r_2=evaluation(orginal_y_test_pred,orginal_y_test_tensor)
#print(f"{config.save_path}:MAE={mae},RMSE={rmse},MAPE={mape},R_2={r_2}")
#mae_, rmse_, mape_, r_2_=diff_evaluation(orginal_y_test_pred,orginal_y_test_tensor)
#print(f"Diff_1:{config.save_path}:MAE={mae_},RMSE={rmse_},MAPE={mape_},R_2={r_2_}")

