import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

_dir = os.path.dirname(os.path.abspath(__file__))

setup(
    name="flash",
    ext_modules=[
        CUDAExtension(
            name="flash",
            sources=[
                "src/flash_if.cu",
            ],
            extra_compile_args={
                "cxx": ["-O2"], 
                "nvcc": ["-O2"]
            },
            include_dirs=[os.path.join(_dir, "3rd/cutlass/include")]
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)