NOTE: currently not frozen so you can do model.model = train(model.model, ...)
@dataclass
class ModelWithMeta:
    meta: Dict
    model: eqx.Module

    def save(self, path):
        # TODO: handle buf saving via zipfile or something or separate method
        os.makedirs(path, exist_ok=True)
        self._save_meta(os.path.join(path, 'meta.json'))
        self._save_model(os.path.join(path, 'model.eqx'))

    def _save_model(self, path):
        save_model_state(self.model, path)

    def _save_meta(self, path):
        json.dump(self.meta, open(path, 'w'))

    @classmethod
    def load(cls, path):
        meta = json.load(open(os.path.join(path, 'meta.json')))
        model = load_model_state(os.path.join(path, 'model.eqx'))
        return cls(meta=meta, model=model)

    def __eq__(self, other):
        check_meta = self.meta == other.meta
        check_model = check_identical(self.model, other.model)
        return check_meta & check_meta


def model_maker(fun):
    """This is a decorator that wraps a function that takes some jsonifiable inputs parameters and returns a model.  The
    wrapped function returns a ModelWithMeta instance."""
    sig = inspect.signature(fun)

    @wraps(fun)
    def inner(*args, **kwargs):
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        model = fun(*args, **kwargs)
        meta = bound.arguments
        return ModelWithMeta(model=model, meta=meta)

    return inner

