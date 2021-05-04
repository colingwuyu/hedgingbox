def get_value(dict_json, default_json, *keys):
    ret_value = dict_json
    default_value = default_json
    for level_key in keys:
        default_value = default_value[level_key]
        ret_value = ret_value.get(level_key, default_value)
    return ret_value
