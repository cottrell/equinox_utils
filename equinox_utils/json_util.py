"""
Differentiate between tuples and lists.
"""
import json


class CustomJSONEncoder(json.JSONEncoder):
    def encode(self, obj):
        def hint_tuples(item):
            if isinstance(item, tuple):
                return {'__tuple__': True, 'items': item}
            if isinstance(item, list):
                return [hint_tuples(e) for e in item]
            if isinstance(item, dict):
                return {key: hint_tuples(value) for key, value in item.items()}
            else:
                return item

        return super(CustomJSONEncoder, self).encode(hint_tuples(obj))


def hinted_tuple_hook(obj):
    if '__tuple__' in obj:
        return tuple(obj['items'])
    else:
        return obj


decoder = json.JSONDecoder(object_hook=hinted_tuple_hook)


def dumps(data, **kwargs):
    return json.dumps(data, cls=CustomJSONEncoder, **kwargs)


def dump(data, fp, **kwargs):
    fp.write(dumps(data, **kwargs))
    # NOTE: DOES NOT WORK
    # return json.dump(data, fp, default=CustomJSONEncoder, **kwargs)


def load(fp):
    return decoder.decode(fp.read())


def loads(jsonstring):
    return decoder.decode(jsonstring)
