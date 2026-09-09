r"""ATen 디스패치에 안 보이는 **의미 경계**를 기록한다.

왜 필요한가
-----------
축 계보를 정확한 포트 위에 세워도 두 자리가 남는다. 둘 다 파이썬 레벨에서 일어나고
`TorchDispatchMode` 에는 흔적이 없다.

1. **`repeat_kv(x, n_rep=1)`** — `if n_rep == 1: return hidden_states`. op 가 **아예 발생하지
   않으므로** 반환 텐서의 물리적 생산자가 입력과 같다. 즉 provenance 를 아무리 정확히 모아도
   `n_kv` 와 `n_h` 가 저절로 갈리지 않는다. 이 경계를 못 잡은 채 전치 간선을 열면
   Zamba2 에서 두 축이 한 등가류로 묶인다(`src/axis_classes.py` 헤더의 기록, 그리고
   2026-09-05 에 실제로 재현됐다: 등가류 충돌 0 -> 12).

2. **`Cache.update(key_states, value_states, ...)`** — 우리가 보는 것은 그 안의 `concat`
   뿐이라 어느 쪽이 key 이고 어느 쪽이 value 인지 순서로 역산해야 한다. GLM-5.2 는
   `d_nope+d_rope == d_v == 256` 이라 그 역산이 불가능하다.

그래서 두 곳에만 **좁은** 계측을 건다. 모듈 후크가 아니라 함수/메서드 wrapper 다 --
`Cache` 는 `nn.Module` 이 아닐 수 있고, 부모 attention 후크만으로는 인자를 구별할 수 없다
(외부 검토 2026-09-09).

기록만 한다. 라벨 결정에는 쓰지 않는다.
"""
import contextvars
import inspect
import itertools

# 지금 어떤 의미 이벤트 안에 있는가. 트레이서가 행마다 읽어 남긴다.
CURRENT = contextvars.ContextVar("semantic_ctx", default=None)

_EVENTS = []
_EID = itertools.count(1)


def events():
    return list(_EVENTS)


def reset():
    _EVENTS.clear()


def _tid(t):
    """텐서의 안정 id. 트레이서가 붙여 둔 것을 그대로 쓴다(없으면 None)."""
    tracer = _ACTIVE.get("tracer")
    if tracer is None or t is None:
        return None
    try:
        return tracer.tensor_uid.get(t)
    except (TypeError, AttributeError):
        return None


_ACTIVE = {}


class SemanticWrappers:
    """추적 중에만 두 지점을 감싼다. `with` 를 벗어나면 원상 복구한다."""

    def __init__(self, tracer, model=None):
        self.tracer = tracer
        self.model = model
        self._undo = []

    # ---- repeat_kv ----------------------------------------------------
    def _wrap_repeat_kv(self):
        """`repeat_kv` 를 감싸 **n_rep 과 무관하게** 논리적 경계를 남긴다.

        n_rep > 1 이면 ATen `expand`/`reshape` 이 실제로 돌아 계보가 이어지지만, n_rep == 1
        이면 같은 텐서가 그대로 돌아온다. 그 경우에도 "여기서 역할이 n_kv -> n_h 로 바뀌었다"를
        적어야 나중에 그 자리를 barrier 로 끊을 수 있다.
        """
        import sys
        targets = []
        for name, mod in list(sys.modules.items()):
            if not name.startswith("transformers."):
                continue
            # **`getattr` 을 쓰면 안 된다.** transformers 는 지연 로딩 모듈이라
            # 속성 접근이 서브모듈 import 를 깨우고, 그 과정에서 없는 의존성
            # (torchvision)까지 끌어와 트레이스가 통째로 죽는다(2026-09-09 실측).
            # 이미 로드된 것만 보면 되므로 `__dict__` 를 직접 읽는다.
            fn = getattr(mod, "__dict__", {}).get("repeat_kv")
            if callable(fn) and getattr(fn, "__module__", "").startswith("transformers"):
                targets.append((mod, fn))
        for mod, fn in targets:
            if getattr(fn, "_llmat_wrapped", False):
                continue

            def make(orig):
                def wrapper(hidden_states, n_rep, *a, **kw):
                    out = orig(hidden_states, n_rep, *a, **kw)
                    _EVENTS.append({
                        "event_id": next(_EID),
                        "kind": "repeat_kv",
                        "n_rep": int(n_rep) if isinstance(n_rep, int) else None,
                        # n_rep == 1 이면 같은 텐서다 -- 그것이 이 이벤트가 존재하는 이유.
                        "noop": out is hidden_states,
                        "in_tensor_id": _tid(hidden_states),
                        "out_tensor_id": _tid(out),
                        "axis": 1,               # [B, n_kv, T, D] 의 축 1
                        "role_before": "n_kv",
                        "role_after": "n_h",
                    })
                    return out
                wrapper._llmat_wrapped = True
                wrapper.__wrapped__ = orig
                return wrapper

            setattr(mod, "repeat_kv", make(fn))
            self._undo.append((mod, "repeat_kv", fn))

    # ---- Cache.update -------------------------------------------------
    def _wrap_cache_update(self):
        """`Cache.update` 를 감싸 key/value 인자 위치를 **이름으로** 확정한다.

        `inspect.signature(...).bind_partial()` 로 `key_states` / `value_states` 를 찾는다.
        위치 0/1 폴백은 쓰지 않는다 -- 이름을 못 찾으면 추정하지 않고 unknown 으로 둔다
        (틀린 provenance 는 없는 provenance 보다 나쁘다).
        """
        try:
            from transformers import cache_utils
        except ImportError:
            return
        seen = set()
        for attr in list(vars(cache_utils)):          # dir() 은 지연 속성을 깨울 수 있다
            cls = vars(cache_utils).get(attr)
            if not isinstance(cls, type) or "update" not in cls.__dict__:
                continue
            orig = cls.__dict__["update"]
            if getattr(orig, "_llmat_wrapped", False) or orig in seen:
                continue
            seen.add(orig)
            try:
                sig = inspect.signature(orig)
            except (TypeError, ValueError):
                continue
            if "key_states" not in sig.parameters or "value_states" not in sig.parameters:
                continue

            def make(orig=orig, sig=sig):
                def wrapper(self, *a, **kw):
                    try:
                        bound = sig.bind_partial(self, *a, **kw)
                        k = bound.arguments.get("key_states")
                        v = bound.arguments.get("value_states")
                    except TypeError:
                        k = v = None
                    eid = next(_EID)
                    _EVENTS.append({
                        "event_id": eid,
                        "kind": "cache_update",
                        "cls": type(self).__name__,
                        "key_tensor_id": _tid(k),
                        "value_tensor_id": _tid(v),
                        "resolved": k is not None and v is not None,
                    })
                    token = CURRENT.set({"event_id": eid, "kind": "cache_update"})
                    try:
                        out = orig(self, *a, **kw)
                    finally:
                        CURRENT.reset(token)
                    # 돌아온 key/value 에도 역할을 다시 붙인다 -- 캐시를 거친 뒤의 텐서가
                    # 이후 attention 으로 들어가므로, 그쪽 계보에도 근거가 있어야 한다.
                    if isinstance(out, (tuple, list)) and len(out) == 2:
                        _EVENTS.append({
                            "event_id": next(_EID), "kind": "cache_update_out",
                            "of_event": eid,
                            "key_tensor_id": _tid(out[0]), "value_tensor_id": _tid(out[1]),
                        })
                    return out
                wrapper._llmat_wrapped = True
                wrapper.__wrapped__ = orig
                return wrapper

            setattr(cls, "update", make())
            self._undo.append((cls, "update", orig))

    def __enter__(self):
        _ACTIVE["tracer"] = self.tracer
        reset()
        self._wrap_repeat_kv()
        self._wrap_cache_update()
        return self

    def __exit__(self, *exc):
        for owner, name, orig in reversed(self._undo):
            setattr(owner, name, orig)
        self._undo.clear()
        _ACTIVE.pop("tracer", None)
        return False
