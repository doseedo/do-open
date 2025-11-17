import multiprocessing as mp
mp.set_start_method('spawn', force=True)

from celery.bin.celery import main
main()
