import json


def pretty_print_json(json_data):
    print(json.dumps(json_data, indent=4))


def log_error(error):
    import traceback

    print(error)
    traceback.print_exc()
