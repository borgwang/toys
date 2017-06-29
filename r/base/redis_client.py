from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import redis
import threading

class RedisClientPool(object):
    _client_pool_dict = dict()
    _mutex = threading.RLock()

    @staticmethod
    def redis_client(redis_url):
        client_pool = RedisClientPool._client_pool_dict.get(redis_url, None)
        if client_pool is None:
            with RedisClientPool._mutex:
                client_pool = RedisClientPool._client_pool_dict.get(redis_url, None)
                if client_pool is None:
                    client_pool = RedisClientPool(redis_url)
                    RedisClientPool._client_pool_dict[redis_url] = client_pool
        return client_pool.create_client()

    def __init__(self, redis_url):
        self.__redis_url = redis_url
        self.__redis_connection_pool = redis.ConnectionPool.from_url(redis_url)

    def __del__(self):
        self.__redis_connection_pool.disconnect()
        RedisClientPool._client_pool_dict.pop(self.__redis_url, None)

    def create_client(self):
        return RedisClient(connection_pool=self.__redis_connection_pool)


class RedisClient(object):
    def __init__(self, connection_pool):
        self.__redis_client = redis.Redis(connection_pool=connection_pool)

    
