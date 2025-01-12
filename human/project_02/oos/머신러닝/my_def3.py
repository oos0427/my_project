#########################################################################################

def get_model_import_by_index(index):
    """
    주어진 인덱스를 기반으로 해당 모델을 동적으로 import하고 반환하는 함수.

    Args:
        index (int): 모델을 선택할 인덱스.
            0 - LinearRegression
            1 - RandomForestRegressor
            2 - GradientBoostingRegressor
            3 - LGBMRegressor
            4 - SVR
            5 - MLPRegressor
            6 - ElasticNet

    Returns:
        module: 선택된 모델 클래스.
    """
    if index == 0:
        from sklearn.linear_model import LinearRegression
        return LinearRegression
    elif index == 1:
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor
    elif index == 2:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor
    elif index == 3:
        from lightgbm import LGBMRegressor
        return LGBMRegressor
    elif index == 4:
        from sklearn.svm import SVR
        return SVR
    elif index == 5:
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor
    elif index == 6:
        from sklearn.linear_model import ElasticNet
        return ElasticNet
    else:
        raise ValueError("유효하지 않은 인덱스입니다. 0~6 사이의 값을 입력하세요.")

#########################################################################################

def reduce_memory_usage(df):
    """
    데이터프레임의 메모리 사용량을 줄이기 위해 데이터 타입을 축소합니다.
    오버플로우를 방지하기 위해 각 열의 값 범위를 확인하여 적절한 데이터 타입으로 변환합니다.

    Parameters:
        df (pd.DataFrame): 데이터 타입 축소를 적용할 데이터프레임.

    Returns:
        pd.DataFrame: 데이터 타입이 축소된 데이터프레임.
    """
    import pandas as pd
    import numpy as np
    
    for col in df.columns:
        col_type = df[col].dtype

        if pd.api.types.is_integer_dtype(col_type):
            # 정수형 처리
            col_min = df[col].min()
            col_max = df[col].max()

            if col_min >= -32768 and col_max <= 32767:
                df[col] = df[col].astype("int16")
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df[col] = df[col].astype("int32")
            # 값이 더 클 경우, int64 유지

        elif pd.api.types.is_float_dtype(col_type):
            # 부동소수점 처리
            col_min = df[col].min()
            col_max = df[col].max()

            if col_max - col_min < 65504:
                df[col] = df[col].astype("float16")
            else:
                df[col] = df[col].astype("float32")
            # float64는 유지되지 않도록 float32로 축소

        elif pd.api.types.is_object_dtype(col_type):
            # 문자열 형식은 그대로 유지
            continue

    print(df.info())

    return df

#########################################################################################

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    모델 평가 함수
    
    전달인자 설명:
    model: 학습된 머신러닝 모델
    X_train: 학습 데이터 피처
    X_test: 테스트 데이터 피처
    y_train: 학습 데이터 타겟
    y_test: 테스트 데이터 타겟
    """
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import numpy as np

    # 테스트 데이터에 대한 예측 수행
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # 평가
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    # 평균값 계산
    train_mean = y_train.mean()
    test_mean = y_test.mean()

    # MAE 대비 비율 계산
    train_mae_ratio = (train_mae / train_mean) * 100
    test_mae_ratio = (test_mae / test_mean) * 100

    # 평가 결과 반환
    return {
        'train_mse': round(train_mse, 2),
        'test_mse': round(test_mse, 2),
        'train_rmse': round(train_rmse, 2),
        'test_rmse': round(test_rmse, 2),
        'train_r2': round(train_r2, 2),
        'test_r2': round(test_r2, 2),
        'train_mae': round(train_mae, 2),
        'test_mae': round(test_mae, 2),
        'train_mae_ratio': round(train_mae_ratio, 2),
        'test_mae_ratio': round(test_mae_ratio, 2)
    }

#########################################################################################

def category_model_evaluation(
    df, 
    model_index, 
    features=None, 
    target=None, 
    category_column="업종별카테고리", 
    test_size=0.2, 
    random_state=42, 
    scaler_type=None, 
    model_params=None
):
    """
    업종별 데이터를 처리하고 단일 모델을 학습하여 성능 평가 결과를 반환합니다.

    Parameters:
        df (pd.DataFrame): 입력 데이터프레임
        model_index (int): 사용할 모델 인덱스.
            0 - LinearRegression
            1 - RandomForestRegressor
            2 - GradientBoostingRegressor
            3 - LGBMRegressor
            4 - SVR
            5 - MLPRegressor
            6 - ElasticNet
        features (list, optional): 사용할 피처 리스트. 기본값은 타겟 열을 제외한 모든 열.
        target (str, optional): 타겟 열 이름. 기본값은 '월매출(점포)'.
        category_column (str, optional): 카테고리를 구분할 열 이름. 기본값은 '업종별카테고리'.
        test_size (float): 테스트 데이터 비율 (기본값: 0.2)
        random_state (int): 랜덤 시드 값 (기본값: 42)
        scaler_type (str): 사용할 스케일러 종류 ('standard', 'minmax', None) (기본값: None)
        model_params (dict): 모델의 하이퍼파라미터 설정 (기본값: None).

    Returns:
        pd.DataFrame: 성능 평가 결과를 담은 데이터프레임.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import pandas as pd

    # 기본 타겟 설정
    default_target = '월매출(점포)'
    target = target or default_target

    # 기본 피처 설정 (타겟 열을 제외한 모든 열)
    features = features or [col for col in df.columns if col != target]

    # 스케일러 초기화
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = None

    # 모델 가져오기
    model_class = get_model_import_by_index(model_index)
    if model_class is None:
        raise ValueError("유효하지 않은 모델 인덱스입니다. 0~6 사이의 값을 입력하세요.")
    model_name = model_class.__name__

    # 모델 초기화
    model = model_class(**(model_params or {}))

    # 카테고리 고유값 추출
    unique_categories = df[category_column].unique()

    # 결과를 저장할 리스트 생성
    results = []

    # 각 카테고리별로 데이터를 나누어 모델 학습 및 예측 수행
    for category in unique_categories:
        # 카테고리별 데이터 필터링
        df_category = df[df[category_column] == category]

        # 문자열 데이터 원핫인코딩
        string_columns = [col for col in features if df_category[col].dtype == 'object']
        df_category = pd.get_dummies(df_category, columns=string_columns, drop_first=True)

        # 원핫 인코딩된 열 추가
        encoded_columns = [col for col in df_category.columns if any(col.startswith(f'{s}_') for s in string_columns)]
        full_features = [col for col in features if col not in string_columns] + encoded_columns

        # 피처와 타겟 데이터 분리
        X = df_category[full_features]
        y = df_category[target]

        # 스케일러 적용 (선택적으로)
        if scaler is not None:
            X = scaler.fit_transform(X)

        # 학습 데이터와 테스트 데이터 분리
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        # 모델 학습
        model.fit(X_train, y_train)

        # 성능 평가 호출 (evaluate_model은 별도 정의된 함수로 가정)
        evaluation_result = evaluate_model(model, X_train, X_test, y_train, y_test)

        # 결과 저장
        if evaluation_result is not None:
            evaluation_result['category'] = category
            results.append(evaluation_result)

    # 결과를 데이터프레임으로 변환
    results_df = pd.DataFrame(results)

    # 카테고리를 인덱스로 설정
    if 'category' in results_df.columns:
        results_df.set_index('category', inplace=True)
        results_df.index.name = None  # 인덱스 이름 제거

    # test_mae_ratio의 최고, 최저, 평균 추가
    test_mae_ratio_max = results_df['test_mae_ratio'].max()
    test_mae_ratio_min = results_df['test_mae_ratio'].min()
    test_mae_ratio_mean = results_df['test_mae_ratio'].mean()

    # 모든 행에 공통으로 최고, 최저, 평균 추가
    test_mae_summary = {
        'test_mae_ratio_max': test_mae_ratio_max,
        'test_mae_ratio_min': test_mae_ratio_min,
        'test_mae_ratio_mean': test_mae_ratio_mean
    }

    for key, value in test_mae_summary.items():
        results_df[key] = value

    # 결과를 소수점 2자리로 포맷팅
    pd.options.display.float_format = '{:.2f}'.format

    # 결과 반환
    return results_df.sort_values(by="test_mae_ratio")




# 사용예시
"""
# SVR 모델을 선택하여 실행
model_params = {'C': 1.0, 'kernel': 'linear'}  # SVR 모델의 하이퍼파라미터

results_df = category_model_evaluation(
    df=my_dataframe,
    model_index=4,  # SVR 선택
    model_params=model_params,
    scaler_type='standard'
)

print(results_df)
"""