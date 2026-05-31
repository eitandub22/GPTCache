from typing import Any, Callable, List

import cachetools

from gptcache.manager.eviction.base import EvictionBase


def popitem_wrapper(func, wrapper_func, clean_size):
    def wrapper(*args, **kwargs):
        keys = []
        try:
            keys = [func(*args, **kwargs)[0] for _ in range(clean_size)]
        except KeyError:
            pass
        wrapper_func(keys)

    return wrapper


class MemoryCacheEviction(EvictionBase):
    """eviction: Memory Cache

    :param policy: eviction strategy
    :type policy: str
    :param maxsize: the maxsize of cache data
    :type maxsize: int
    :param clean_size: will clean the size of data when the size of cache data reaches the max size
    :type clean_size: int
    :param on_evict: the function for cleaning the data in the store
    :type  on_evict: Callable[[List[Any]], None]


    """

    def __init__(
            self,
            policy: str = "LRU",
            maxsize: int = 1000,
            clean_size: int = 0,
            on_evict: Callable[[List[Any]], None] = None,
            **kwargs,
    ):
        self._policy = policy.upper()
        if self._policy == "LRU":
            self._cache = cachetools.LRUCache(maxsize=maxsize, **kwargs)
        elif self._policy == "LFU":
            self._cache = cachetools.LFUCache(maxsize=maxsize, **kwargs)
        elif self._policy == "FIFO":
            self._cache = cachetools.FIFOCache(maxsize=maxsize, **kwargs)
        elif self._policy == "RR":
            self._cache = cachetools.RRCache(maxsize=maxsize, **kwargs)
        elif self._policy == "CA_W_TINYLFU":
            from gptcache.manager.eviction.ca_w_tinylfu import CostAwareWTinyLFU
            self._cache = CostAwareWTinyLFU(
                maxsize=maxsize,
                clean_size=clean_size,
                on_evict=on_evict,
                **kwargs,
            )
        else:
            raise ValueError(f"Unknown policy {policy}")

        if self._policy != "CA_W_TINYLFU":
            self._cache.popitem = popitem_wrapper(self._cache.popitem, on_evict, clean_size)

    def put(self, objs: List[Any]):
        if self._policy == "CA_W_TINYLFU":
            self._cache.put(objs)
        else:
            for obj in objs:
                self._cache[obj] = True

    def get(self, obj: Any):
        return self._cache.get(obj)

    @property
    def policy(self) -> str:
        return self._policy
