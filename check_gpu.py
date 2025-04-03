import tensorflow as tf

def check_gpu():
    print("TensorFlow version:", tf.__version__)
    print("\nGPU Available:", tf.config.list_physical_devices('GPU'))
    
    # Get GPU device name if available
    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print("\nGPU Device Name:", tf.test.gpu_device_name())
    else:
        print("\nNo GPU devices found. TensorFlow is using CPU only.")
    
    # Print memory info if GPU is available
    if gpu_devices:
        try:
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            print("\nGPU Memory Info:")
            print("Current: {:.2f} GB".format(memory_info['current'] / 1024**3))
            print("Peak: {:.2f} GB".format(memory_info['peak'] / 1024**3))
        except:
            print("\nCould not get GPU memory information")

if __name__ == "__main__":
    check_gpu() 