import os

# las matrices del LETKF son chicas (N x N): los hilos de BLAS solo estorban
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
