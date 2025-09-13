from typing import Iterable, List, Tuple, Any, Dict


def split_block_keep_order(objs: List[Any], min_chunk_size: int = 8) -> List[List[Any]]:
    """
    Split a list 'objs' into chunks, each of length >= min_chunk_size.
    Order is preserved. If the list is too short to split into two valid chunks,
    it is returned as a single block.
    """
    n = len(objs)
    # If we can't make at least two chunks of size >= min_chunk_size, keep as one block
    if n < 2 * min_chunk_size:
        return [objs]

    # Number of chunks if we start from min sizes
    k = n // min_chunk_size  # at least 2 here because of the guard above
    sizes = [min_chunk_size] * k
    remaining = n - min_chunk_size * k

    # Distribute the remainder as evenly as possible across chunks (preserving order)
    i = 0
    while remaining > 0:
        sizes[i % k] += 1
        remaining -= 1
        i += 1

    # Slice according to computed sizes
    chunks = []
    start = 0
    for s in sizes:
        chunks.append(objs[start:start + s])
        start += s
    return chunks


def cluster_blocks_with_min_size(
    pairs: Dict[Any, int],
    min_chunk_size: int = 8
) -> List[List[Any]]:
    """
    pairs: iterable of (cluster_id, obj), in the original order.
    Returns: list of lists of objects, where each sublist is a consecutive block
             of the same cluster_id. Long blocks are split into chunks with
             each chunk size >= min_chunk_size.
    """
    result: List[List[Any]] = []
    current_cluster = None
    current_objs: List[Any] = []

    for obj, cluster_id in pairs.items():
        if current_cluster is None:
            current_cluster = cluster_id
            current_objs = [obj]
        elif cluster_id == current_cluster:
            current_objs.append(obj)
        else:
            # finish previous block; split if needed
            result.extend(split_block_keep_order(current_objs, min_chunk_size))
            current_cluster = cluster_id
            current_objs = [obj]

    if current_objs:
        result.extend(split_block_keep_order(current_objs, min_chunk_size))

    return result