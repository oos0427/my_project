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

def linear_category_model(df, features=None, target=None, category_column="업종별카테고리", test_size=0.2, random_state=42, scaler_type=None):
    """
    카테고리별 데이터를 처리하고 선형 회귀 모델을 학습하여 성능 평가 결과를 반환합니다.

    Parameters:
        df (pd.DataFrame): 입력 데이터프레임
        features (list, optional): 사용할 피처 리스트. 기본값은 지정된 기본 피처.
        target (str, optional): 타겟 열 이름. 기본값은 '월매출(점포)'.
        category_column (str, optional): 카테고리를 구분할 열 이름. 기본값은 '업종별카테고리'.
        test_size (float): 테스트 데이터 비율 (기본값: 0.2)
        random_state (int): 랜덤 시드 값 (기본값: 42)
        scaler_type (str): 사용할 스케일러 종류 ('standard', 'minmax', None) (기본값: None)

    Returns:
        pd.DataFrame: 성능 평가 결과를 담은 데이터프레임
    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import pandas as pd

    # 기본 피처 및 타겟 설정
    default_features = ['년분기', '인구수', '지역생활인구', '장기외국인', '단기외국인', '행정동',
                        '주차장면적(면)', '주차장개수(개소)', '학교수', '학생수', '버스정류장수']
    default_target = '월매출(점포)'

    features = features or default_features
    target = target or default_target

    # 스케일러 초기화
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = None

    # 카테고리 고유값 추출
    unique_categories = df[category_column].unique()

    # 결과를 저장할 딕셔너리 생성
    results = {}

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

        # 선형 회귀 모델 학습
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 성능 평가 호출 (evaluate_model은 별도 정의된 함수로 가정)
        evaluation_result = evaluate_model(model, X_train, X_test, y_train, y_test)

        # 결과 저장
        if evaluation_result is not None:
            results[category] = evaluation_result

    # 결과를 데이터프레임으로 변환
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df = results_df.apply(pd.to_numeric)

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

#########################################################################################

def random_forest_category_model(df, features=None, target=None, category_column="업종별카테고리", test_size=0.2, random_state=42, scaler_type=None, model_params=None):
    """
    업종별 데이터를 처리하고 랜덤 포레스트 회귀 모델을 학습하여 성능 평가 결과를 반환합니다.

    Parameters:
        df (pd.DataFrame): 입력 데이터프레임
        features (list, optional): 사용할 피처 리스트. 기본값은 지정된 기본 피처.
        target (str, optional): 타겟 열 이름. 기본값은 '월매출(점포)'.
        category_column (str, optional): 카테고리를 구분할 열 이름. 기본값은 '업종별카테고리'.
        test_size (float): 테스트 데이터 비율 (기본값: 0.2)
        random_state (int): 랜덤 시드 값 (기본값: 42)
        scaler_type (str): 사용할 스케일러 종류 ('standard', 'minmax', None) (기본값: None)
        model_params (dict): RandomForestRegressor의 하이퍼파라미터 설정 (기본값: None)

    Returns:
        pd.DataFrame: 성능 평가 결과를 담은 데이터프레임
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import pandas as pd

    # 기본 피처 및 타겟 설정
    default_features = ['년분기', '인구수', '지역생활인구', '장기외국인', '단기외국인', '행정동',
                        '주차장면적(면)', '주차장개수(개소)', '학교수', '학생수', '버스정류장수']
    default_target = '월매출(점포)'

    features = features or default_features
    target = target or default_target

    # 스케일러 초기화
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = None

    # 카테고리 고유값 추출
    unique_categories = df[category_column].unique()

    # 결과를 저장할 딕셔너리 생성
    results = {}

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

        # 랜덤 포레스트 회귀 모델 학습
        model = RandomForestRegressor(random_state=random_state, **(model_params or {}))
        model.fit(X_train, y_train)

        # 성능 평가 호출 (evaluate_model은 별도 정의된 함수로 가정)
        evaluation_result = evaluate_model(model, X_train, X_test, y_train, y_test)

        # 결과 저장
        if evaluation_result is not None:
            results[category] = evaluation_result

    # 결과를 데이터프레임으로 변환
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df = results_df.apply(pd.to_numeric)

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

#########################################################################################

def gradient_boosting_category_model(df, features=None, target=None, category_column="업종별카테고리", test_size=0.2, random_state=42, scaler_type=None, model_params=None):
    """
    업종별 데이터를 처리하고 Gradient Boosting 회귀 모델을 학습하여 성능 평가 결과를 반환합니다.

    Parameters:
        df (pd.DataFrame): 입력 데이터프레임
        features (list, optional): 사용할 피처 리스트. 기본값은 지정된 기본 피처.
        target (str, optional): 타겟 열 이름. 기본값은 '월매출(점포)'.
        category_column (str, optional): 카테고리를 구분할 열 이름. 기본값은 '업종별카테고리'.
        test_size (float): 테스트 데이터 비율 (기본값: 0.2)
        random_state (int): 랜덤 시드 값 (기본값: 42)
        scaler_type (str): 사용할 스케일러 종류 ('standard', 'minmax', None) (기본값: None)
        model_params (dict): GradientBoostingRegressor의 하이퍼파라미터 설정 (기본값: None)

    Returns:
        pd.DataFrame: 성능 평가 결과를 담은 데이터프레임
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import pandas as pd

    # 기본 피처 및 타겟 설정
    default_features = ['년분기', '인구수', '지역생활인구', '장기외국인', '단기외국인', '행정동',
                        '주차장면적(면)', '주차장개수(개소)', '학교수', '학생수', '버스정류장수']
    default_target = '월매출(점포)'

    features = features or default_features
    target = target or default_target

    # 스케일러 초기화
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = None

    # 카테고리 고유값 추출
    unique_categories = df[category_column].unique()

    # 결과를 저장할 딕셔너리 생성
    results = {}

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

        # Gradient Boosting 회귀 모델 학습
        model = GradientBoostingRegressor(random_state=random_state, **(model_params or {}))
        model.fit(X_train, y_train)

        # 성능 평가 호출 (evaluate_model은 별도 정의된 함수로 가정)
        evaluation_result = evaluate_model(model, X_train, X_test, y_train, y_test)

        # 결과 저장
        if evaluation_result is not None:
            results[category] = evaluation_result

    # 결과를 데이터프레임으로 변환
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df = results_df.apply(pd.to_numeric)

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

#########################################################################################

def lgbm_category_model(df, features=None, target=None, category_column="업종별카테고리", test_size=0.2, random_state=42, scaler_type=None, model_params=None):
    """
    업종별 데이터를 처리하고 LGBM 회귀 모델을 학습하여 성능 평가 결과를 반환합니다.

    Parameters:
        df (pd.DataFrame): 입력 데이터프레임
        features (list, optional): 사용할 피처 리스트. 기본값은 지정된 기본 피처.
        target (str, optional): 타겟 열 이름. 기본값은 '월매출(점포)'.
        category_column (str, optional): 카테고리를 구분할 열 이름. 기본값은 '업종별카테고리'.
        test_size (float): 테스트 데이터 비율 (기본값: 0.2)
        random_state (int): 랜덤 시드 값 (기본값: 42)
        scaler_type (str): 사용할 스케일러 종류 ('standard', 'minmax', None) (기본값: None)
        model_params (dict): LGBMRegressor의 하이퍼파라미터 설정 (기본값: None)

    Returns:
        pd.DataFrame: 성능 평가 결과를 담은 데이터프레임
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from lightgbm import LGBMRegressor
    import pandas as pd

    # 기본 피처 및 타겟 설정
    default_features = ['년분기', '인구수', '지역생활인구', '장기외국인', '단기외국인', '행정동',
                        '주차장면적(면)', '주차장개수(개소)', '학교수', '학생수', '버스정류장수']
    default_target = '월매출(점포)'

    features = features or default_features
    target = target or default_target

    # 스케일러 초기화
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = None

    # 카테고리 고유값 추출
    unique_categories = df[category_column].unique()

    # 결과를 저장할 딕셔너리 생성
    results = {}

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

        # LGBM 회귀 모델 학습
        model = LGBMRegressor(random_state=random_state, **(model_params or {}))
        model.fit(X_train, y_train)

        # 성능 평가 호출 (evaluate_model은 별도 정의된 함수로 가정)
        evaluation_result = evaluate_model(model, X_train, X_test, y_train, y_test)

        # 결과 저장
        if evaluation_result is not None:
            results[category] = evaluation_result

    # 결과를 데이터프레임으로 변환
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df = results_df.apply(pd.to_numeric)

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

#########################################################################################

def svr_category_model(df, features=None, target=None, category_column="업종별카테고리", test_size=0.2, random_state=42, scaler_type=None, model_params=None):
    """
    업종별 데이터를 처리하고 SVR 모델을 학습하여 성능 평가 결과를 반환합니다.

    Parameters:
        df (pd.DataFrame): 입력 데이터프레임
        features (list, optional): 사용할 피처 리스트. 기본값은 지정된 기본 피처.
        target (str, optional): 타겟 열 이름. 기본값은 '월매출(점포)'.
        category_column (str, optional): 카테고리를 구분할 열 이름. 기본값은 '업종별카테고리'.
        test_size (float): 테스트 데이터 비율 (기본값: 0.2)
        random_state (int): 랜덤 시드 값 (기본값: 42)
        scaler_type (str): 사용할 스케일러 종류 ('standard', 'minmax', None) (기본값: None)
        model_params (dict): SVR의 하이퍼파라미터 설정 (기본값: None)

    Returns:
        pd.DataFrame: 성능 평가 결과를 담은 데이터프레임
    """
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import pandas as pd

    # 기본 피처 및 타겟 설정
    default_features = ['년분기', '인구수', '지역생활인구', '장기외국인', '단기외국인', '행정동',
                        '주차장면적(면)', '주차장개수(개소)', '학교수', '학생수', '버스정류장수']
    default_target = '월매출(점포)'

    features = features or default_features
    target = target or default_target

    # 스케일러 초기화
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = None

    # 카테고리 고유값 추출
    unique_categories = df[category_column].unique()

    # 결과를 저장할 딕셔너리 생성
    results = {}

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

        # SVR 모델 학습
        model = SVR(**(model_params or {}))
        model.fit(X_train, y_train)

        # 성능 평가 호출 (evaluate_model은 별도 정의된 함수로 가정)
        evaluation_result = evaluate_model(model, X_train, X_test, y_train, y_test)

        # 결과 저장
        if evaluation_result is not None:
            results[category] = evaluation_result

    # 결과를 데이터프레임으로 변환
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df = results_df.apply(pd.to_numeric)

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

#########################################################################################

def mlp_category_model(df, features=None, target=None, category_column="업종별카테고리", test_size=0.2, random_state=42, scaler_type=None, model_params=None):
    """
    업종별 데이터를 처리하고 MLP 회귀 모델을 학습하여 성능 평가 결과를 반환합니다.

    Parameters:
        df (pd.DataFrame): 입력 데이터프레임
        features (list, optional): 사용할 피처 리스트. 기본값은 지정된 기본 피처.
        target (str, optional): 타겟 열 이름. 기본값은 '월매출(점포)'.
        category_column (str, optional): 카테고리를 구분할 열 이름. 기본값은 '업종별카테고리'.
        test_size (float): 테스트 데이터 비율 (기본값: 0.2)
        random_state (int): 랜덤 시드 값 (기본값: 42)
        scaler_type (str): 사용할 스케일러 종류 ('standard', 'minmax', None) (기본값: None)
        model_params (dict): MLPRegressor의 하이퍼파라미터 설정 (기본값: None)

    Returns:
        pd.DataFrame: 성능 평가 결과를 담은 데이터프레임
    """
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import pandas as pd

    # 기본 피처 및 타겟 설정
    default_features = ['년분기', '인구수', '지역생활인구', '장기외국인', '단기외국인', '행정동',
                        '주차장면적(면)', '주차장개수(개소)', '학교수', '학생수', '버스정류장수']
    default_target = '월매출(점포)'

    features = features or default_features
    target = target or default_target

    # 스케일러 초기화
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = None

    # 카테고리 고유값 추출
    unique_categories = df[category_column].unique()

    # 결과를 저장할 딕셔너리 생성
    results = {}

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

        # MLP 회귀 모델 학습
        model = MLPRegressor(random_state=random_state, **(model_params or {}))
        model.fit(X_train, y_train)

        # 성능 평가 호출 (evaluate_model은 별도 정의된 함수로 가정)
        evaluation_result = evaluate_model(model, X_train, X_test, y_train, y_test)

        # 결과 저장
        if evaluation_result is not None:
            results[category] = evaluation_result

    # 결과를 데이터프레임으로 변환
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df = results_df.apply(pd.to_numeric)

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

#########################################################################################

def elasticnet_category_model(df, features=None, target=None, category_column="업종별카테고리", test_size=0.2, random_state=42, scaler_type=None, model_params=None):
    """
    업종별 데이터를 처리하고 ElasticNet 회귀 모델을 학습하여 성능 평가 결과를 반환합니다.

    Parameters:
        df (pd.DataFrame): 입력 데이터프레임
        features (list, optional): 사용할 피처 리스트. 기본값은 지정된 기본 피처.
        target (str, optional): 타겟 열 이름. 기본값은 '월매출(점포)'.
        category_column (str, optional): 카테고리를 구분할 열 이름. 기본값은 '업종별카테고리'.
        test_size (float): 테스트 데이터 비율 (기본값: 0.2)
        random_state (int): 랜덤 시드 값 (기본값: 42)
        scaler_type (str): 사용할 스케일러 종류 ('standard', 'minmax', None) (기본값: None)
        model_params (dict): ElasticNet의 하이퍼파라미터 설정 (기본값: None)

    Returns:
        pd.DataFrame: 성능 평가 결과를 담은 데이터프레임
    """
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import ElasticNet
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    import pandas as pd

    # 기본 피처 및 타겟 설정
    default_features = ['년분기', '인구수', '지역생활인구', '장기외국인', '단기외국인', '행정동',
                        '주차장면적(면)', '주차장개수(개소)', '학교수', '학생수', '버스정류장수']
    default_target = '월매출(점포)'

    features = features or default_features
    target = target or default_target

    # 스케일러 초기화
    if scaler_type == 'standard':
        scaler = StandardScaler()
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = None

    # 카테고리 고유값 추출
    unique_categories = df[category_column].unique()

    # 결과를 저장할 딕셔너리 생성
    results = {}

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

        # ElasticNet 모델 학습
        model = ElasticNet(random_state=random_state, **(model_params or {}))
        model.fit(X_train, y_train)

        # 성능 평가 호출 (evaluate_model은 별도 정의된 함수로 가정)
        evaluation_result = evaluate_model(model, X_train, X_test, y_train, y_test)

        # 결과 저장
        if evaluation_result is not None:
            results[category] = evaluation_result

    # 결과를 데이터프레임으로 변환
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df = results_df.apply(pd.to_numeric)

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

#########################################################################################