def int_multiplier(discretize_size: float):
    int_mul = 1
    x_int_code = discretize_size

    while (x_int_code - int(x_int_code)) > 1e-5:
        int_mul *= 10
        x_int_code = x_int_code*10

    return int_mul
