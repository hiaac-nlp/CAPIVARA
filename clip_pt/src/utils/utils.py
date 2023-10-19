import math


def compute_n_batches(dataset_size, batch_size, shard_size=10000):
    batches_per_shards = math.ceil(shard_size / batch_size)
    batches_in_last_shard = math.ceil((dataset_size % shard_size) / batch_size)
    n_shards = dataset_size // shard_size
    print("n_shards", n_shards)
    print("batches_per_shards", batches_per_shards)
    print("batches_in_last_shard", batches_in_last_shard)
    return n_shards * batches_per_shards + batches_in_last_shard
