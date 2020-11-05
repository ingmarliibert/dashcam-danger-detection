## Trouble shooting

On my manjaro laptop:
```
sudo pacman -Syu tensorflow-cuda cuda cudnn python-pycuda python-tensorflow-cuda python-matplotlib
```

https://github.com/tensorflow/tensorflow/issues/39132

```
Could not load dynamic library 'libcudart.so.10.1'; dlerror: libcudart.so.10.1: cannot open shared object file: No such file or directory
```

Fix:
```
sudo ln -s /opt/cuda/targets/x86_64-linux/lib/libcudart.so /usr/lib/libcudart.so.10.1
```

You do the same for every missing file.
