import os
from typing import List, Optional, Union

import numpy as np

from gptcache.manager.vector_data.base import VectorBase, VectorData
from gptcache.utils import import_faiss
from gptcache.utils.log import gptcache_log

import_faiss()

import faiss  # pylint: disable=C0413


class Faiss(VectorBase):
    """vector store: Faiss

    Supports multiple index types for different performance trade-offs:

    - ``"flat"`` (default): Exact brute-force search with ``IDMap,Flat``.
      Best recall, O(n) search. No training needed.
    - ``"hnsw_sq8"``: HNSW graph index with 8-bit scalar quantization,
      wrapped in ``IndexIDMap`` for custom ID support.
      ~4x memory reduction vs Flat, O(log n) search, high recall.
      **Does not support per-vector deletion** — uses tombstone marking
      and periodic rebuild instead.

    :param index_path: the path to Faiss index, defaults to 'faiss.index'.
    :type index_path: str
    :param dimension: the dimension of the vector, defaults to 0.
    :type dimension: int
    :param top_k: the number of the vectors results to return, defaults to 1.
    :type top_k: int
    :param index_type: index type, one of ``"flat"`` or ``"hnsw_sq8"``, defaults to ``"flat"``.
    :type index_type: str
    :param hnsw_m: number of links per node in HNSW graph (higher = better recall, more memory),
                   defaults to 32.
    :type hnsw_m: int
    :param hnsw_ef_construction: size of the dynamic candidate list during construction
                                 (higher = better recall, slower build), defaults to 200.
    :type hnsw_ef_construction: int
    :param hnsw_ef_search: size of the dynamic candidate list during search
                           (higher = better recall, slower search), defaults to 128.
    :type hnsw_ef_search: int
    """

    def __init__(
        self,
        index_file_path,
        dimension,
        top_k,
        index_type="flat",
        hnsw_m=32,
        hnsw_ef_construction=200,
        hnsw_ef_search=128,
    ):
        self._index_file_path = index_file_path
        self._dimension = dimension
        self._top_k = top_k
        self._index_type = index_type.lower()
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._hnsw_ef_search = hnsw_ef_search

        # For HNSW: tombstone set of deleted IDs (since HNSW can't do remove_ids)
        self._tombstones = set()

        if os.path.isfile(index_file_path):
            self._index = faiss.read_index(index_file_path)
            # Restore tombstones if saved alongside the index
            tombstone_path = index_file_path + ".tombstones.npy"
            if os.path.isfile(tombstone_path):
                self._tombstones = set(np.load(tombstone_path).tolist())
            gptcache_log.info(
                "Loaded existing Faiss index from %s (ntotal=%d, tombstones=%d)",
                index_file_path,
                self._index.ntotal,
                len(self._tombstones),
            )
        else:
            self._index = self._create_index(
                dimension, self._index_type, hnsw_m, hnsw_ef_construction, hnsw_ef_search
            )

    @staticmethod
    def _create_index(dimension, index_type, hnsw_m=32, hnsw_ef_construction=200, hnsw_ef_search=128):
        """Create a new FAISS index of the specified type.

        :param dimension: vector dimensionality.
        :param index_type: ``"flat"`` or ``"hnsw_sq8"``.
        :param hnsw_m: HNSW M parameter.
        :param hnsw_ef_construction: HNSW efConstruction parameter.
        :param hnsw_ef_search: HNSW efSearch parameter.
        :return: a configured ``faiss.Index`` wrapped in ``IndexIDMap``.
        """
        if index_type == "hnsw_sq8":
            # HNSW graph with 8-bit Scalar Quantization
            # - HNSW{M}: graph connectivity (higher M = better recall, more memory)
            # - SQ8: each float32 compressed to uint8 (4x memory savings)
            factory_string = f"HNSW{hnsw_m},SQ8"
            base_index = faiss.index_factory(dimension, factory_string, faiss.METRIC_L2)

            # Set HNSW-specific parameters for recall/speed trade-off
            hnsw_index = faiss.downcast_index(base_index)
            hnsw_index.hnsw.efSearch = hnsw_ef_search
            hnsw_index.hnsw.efConstruction = hnsw_ef_construction

            # Wrap in IndexIDMap so we can use add_with_ids (custom IDs)
            # HNSW natively uses sequential IDs; IDMap translates custom → internal
            index = faiss.IndexIDMap(base_index)

            # SQ8 requires a lightweight training step (learns min/max per dimension).
            # Unlike IVF+PQ, this is virtually instant and can be done on the first
            # batch of vectors — no cold start problem.
            gptcache_log.info(
                "Created HNSW+SQ8 index (dim=%d, M=%d, efConstruction=%d, efSearch=%d)",
                dimension, hnsw_m, hnsw_ef_construction, hnsw_ef_search,
            )
            return index
        else:
            # Default: exact brute-force with ID mapping
            return faiss.index_factory(dimension, "IDMap,Flat", faiss.METRIC_L2)

    @property
    def index_type(self):
        return self._index_type

    def mul_add(self, datas: List[VectorData]):
        data_array, id_array = map(list, zip(*((data.data, data.id) for data in datas)))
        np_data = np.array(data_array).astype("float32")
        ids = np.array(id_array)

        if self._index_type == "hnsw_sq8" and not self._index.is_trained:
            # SQ8 training: learns per-dimension min/max for quantization.
            # This is virtually instant (unlike IVF+PQ which needs ~10K vectors).
            self._index.train(np_data)
            gptcache_log.info("Trained HNSW+SQ8 index on %d vectors", len(np_data))

        self._index.add_with_ids(np_data, ids)

    def search(self, data: np.ndarray, top_k: int = -1):
        if self._index.ntotal == 0:
            return None
        if top_k == -1:
            top_k = self._top_k

        np_data = np.array(data).astype("float32").reshape(1, -1)

        if self._index_type == "hnsw_sq8" and self._tombstones:
            # Over-fetch to compensate for tombstoned results we'll filter out
            fetch_k = min(top_k + len(self._tombstones), self._index.ntotal)
            dist, ids = self._index.search(np_data, fetch_k)
            # Filter out tombstoned IDs
            results = []
            for d, i in zip(dist[0], ids[0]):
                i = int(i)
                if i == -1 or i in self._tombstones:
                    continue
                results.append((d, i))
                if len(results) >= top_k:
                    break
            return results if results else None
        else:
            dist, ids = self._index.search(np_data, top_k)
            ids = [int(i) for i in ids[0]]
            return list(zip(dist[0], ids))

    def rebuild(self, ids=None):
        """Rebuild the index, removing physically deleted vectors where possible.

        For flat: clears the (unused) tombstone set — physical removal was
        already done by ``remove_ids`` in ``delete()``.
        For hnsw_sq8: ``IndexHNSWSQ`` does not expose a decode path for stored
        SQ8 codes, so structural compaction is not possible here. Tombstones
        are intentionally **kept** so that evicted IDs continue to be filtered
        out of search results in all subsequent calls to ``search()``.

        The tombstone set is bounded by the total number of evictions over the
        cache lifetime (each eviction adds at most one entry). At 8 bytes per
        int64, even 1 million cumulative evictions costs only ~8 MB.
        """
        if self._index_type == "hnsw_sq8":
            # HNSW cannot physically remove vectors. Tombstones remain active
            # and must NOT be cleared — clearing them would allow deleted
            # vectors to reappear in search results.
            return True
        self._tombstones.clear()
        return True

    def delete(self, ids):
        """Delete vectors by their IDs.

        For Flat index: uses FAISS native ``remove_ids``.
        For HNSW+SQ8: marks IDs as tombstones (logical deletion) since
        HNSW does not support structural deletion. Tombstoned IDs are
        filtered out during search and removed on the next ``rebuild()``.
        """
        if self._index_type == "hnsw_sq8":
            # HNSW does not support remove_ids — use tombstone marking
            self._tombstones.update(int(i) for i in ids)
            gptcache_log.debug(
                "Tombstoned %d IDs in HNSW index (total tombstones: %d)",
                len(ids),
                len(self._tombstones),
            )
        else:
            ids_to_remove = np.array(ids)
            self._index.remove_ids(faiss.IDSelectorBatch(ids_to_remove.size, faiss.swig_ptr(ids_to_remove)))

    def flush(self):
        faiss.write_index(self._index, self._index_file_path)
        tombstone_path = self._index_file_path + ".tombstones.npy"
        if self._tombstones:
            np.save(tombstone_path, np.array(list(self._tombstones)))
        elif os.path.isfile(tombstone_path):
            # Remove stale tombstone file left over from a previous flush so
            # that a subsequent load does not restore already-evicted IDs.
            os.remove(tombstone_path)

    def close(self):
        self.flush()

    def count(self):
        return self._index.ntotal
