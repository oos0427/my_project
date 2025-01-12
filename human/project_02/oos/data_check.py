def Data_Comparison2(dict_to_compare, c):
    """
    dict_to_compare = {"sales" :sales, "store": store }
    dict_to_compare = {"출력명" :데이터저장된 변수, "출력명": 데이터저장된 변수 }
    """

    keys = list(dict_to_compare.keys())

    set_a = set(dict_to_compare[keys[0]][c])
    set_b = set(dict_to_compare[keys[1]][c])
    common_values = set_a & set_b  # A와 B에 모두 있는 값
    only_in_a = set_a - set_b  # A에만 있는 값
    only_in_b = set_b - set_a  # B에만 있는 값

    # 결과 출력
    print("모두 있는 값:")
    print(common_values)

    print(f"\n{keys[0]}에만 있는 값:")
    print(only_in_a)

    print(f"\n{keys[1]}에만 있는 값:")
    print(only_in_b)