import tensorflow as tf
from numba import cuda 

def test():
    # Verify the CPU setup  
    python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"  
    # A tensor should be return, something like  
    # tf.Tensor(-686.383, shape=(), dtype=float32)  
    
    # Verify the GPU setup  
    python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('[GPU](https://saturncloud.io/glossary/gpu)'))"  
    # A list of GPU devices should be return, something like  
    # [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]


def reset_current_gpu():
    """
    Réinitialise le GPU actuellement utilisé.

    Cette fonction récupère le GPU actuellement utilisé et le réinitialise.
    Cela peut être utile pour libérer des ressources ou pour réinitialiser l'état du GPU
    après une utilisation intensive.

    Note : Cette fonction utilise la bibliothèque CUDA pour interagir avec le GPU.
    """
    device = cuda.get_current_device()
    device.reset()

def limitgpu_ram_usage(limit=1024, gpu_indice=0):
    """
    Limite l'utilisation de la RAM du GPU.

    Cette fonction limite l'utilisation de la RAM du premier GPU physique à une valeur spécifiée
    (par défaut, 1024 Mo). Elle utilise TensorFlow pour configurer les dispositifs logiques.

    Paramètres :
    limit (int) : La limite de mémoire en Mo à allouer au premier GPU. Par défaut, 1024 Mo.
    gpu_indice (int) : Indice du GPU à utiliser, par défaut le premier est sélectionné.

    Note : Cette fonction doit être appelée avant que les GPUs ne soient initialisés par TensorFlow.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[gpu_indice],
                [tf.config.LogicalDeviceConfiguration(memory_limit=limit)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
