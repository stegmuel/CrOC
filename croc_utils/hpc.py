import os
import shutil


def signal_handler(args, sig, frame):
    print(args)
    clean_on_leave(args)
    print('clean_on_leave() was called')


def pin_workers_iterator(the_iterator, args):
    try:
        print(args.cpus)
    except AttributeError:
        args.cpus = list(sorted(os.sched_getaffinity(0)))

    if args.num_workers > 0:
        for index, w in enumerate(the_iterator._workers):
            os.system("taskset -p -c %d %d" % ((args.cpus[(index + 1) % len(args.cpus)]), w.pid))


def clean_on_leave(args):
    if args.untar_path[:8] == '/dev/shm' and int(args.gpu) == 0:
        shutil.rmtree(args.untar_path)
