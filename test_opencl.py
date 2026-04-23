import pyopencl as cl


def check_devices():
    platforms = cl.get_platforms()
    if not platforms:
        print("No OpenCL platforms found.")
        return
    
    for platform in platforms:
            print(f"Platform: {platform.name}")
            devices = platform.get_devices()
            for device in devices:
                print(f"  Device: {device.name}")
                print(f"  Max Work Group Size: {device.max_work_group_size}")
                print(f"  Local Memory Size: {device.local_mem_size / 1024} KB")
                print("-" * 30)

if __name__ == "__main__":
    check_devices()