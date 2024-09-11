import tensorflow as tf
import time
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt

# Vérifier si CUDA est disponible
if tf.config.list_physical_devices('GPU'):
    print("CUDA is available")
else:
    print("CUDA is not available")


cpu_times = []
sizes = [1, 10, 100, 500, 1000, 2000, 3000]
for size in sizes:
    print('size:', size)
    ops.reset_default_graph()
    start = time.time()
    with tf.device('cpu:0'):
        v1 = tf.Variable(tf.random.normal((size, size)))
        v2 = tf.Variable(tf.random.normal((size, size)))
        op = tf.matmul(v1, v2)

    tf.print(tf.compat.v1.global_variables_initializer())
    tf.print(op)
    cpu_times.append(time.time() - start)
    print('cpu time took: {0:.4f}'.format(time.time() - start))


gpu_times = []
for size in sizes:
    print('size:', size)
    ops.reset_default_graph()
    start = time.time()
    with tf.device('gpu:0'):
        v1 = tf.Variable(tf.random.normal((size, size)))
        v2 = tf.Variable(tf.random.normal((size, size)))
        op = tf.matmul(v1, v2)

    tf.print(tf.compat.v1.global_variables_initializer())
    tf.print(op)
    gpu_times.append(time.time() - start)
    print('gpu time took: {0:.4f}'.format(time.time() - start))

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(sizes, gpu_times, label='GPU')
ax.plot(sizes, cpu_times, label='CPU')
plt.xlabel('MATRIX SIZE')
plt.ylabel('TIME (sec)')
plt.legend()
plt.show()