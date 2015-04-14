from multiprocessing.pool import ThreadPool
import itertools

__author__ = 'Alex Rogozhnikov, Tatiana Likhomanenko'


def _threads_wrapper(func_and_args):
    func = func_and_args[0]
    args = func_and_args[1:]
    return func(*args)


def map_on_cluster(parallel_profile, *args, **kw_args):
    """
    The same as map, but the first argument is ipc_profile. Distributes the task over IPython cluster.

    :param parallel_profile: the IPython cluster profile to use.
    :type parallel_profile: None or str
    :param list args: function, arguments
    :param dict kw_args: kwargs for LoadBalacedView.map_sync

    :return: the result of mapping
    """
    if parallel_profile is None:
        return map(*args)
    elif str.startswith(parallel_profile, 'threads-'):
        n_threads = int(parallel_profile[len('threads-'):])
        pool = ThreadPool(processes=n_threads)
        func, params = args[0], args[1:]
        return pool.map(_threads_wrapper, zip(itertools.cycle([func]), *params))
    else:
        from IPython.parallel import Client

        return Client(profile=parallel_profile).load_balanced_view().map_sync(*args, **kw_args)