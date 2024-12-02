import time
from functools import wraps
import rospy


def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        rospy.loginfo(f"Function '{func.__name__}' took {elapsed_time:.6f} seconds")
        return result
    return wrapper
