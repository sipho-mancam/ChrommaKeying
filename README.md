# CudaChromaUbuntu

The CudaChromaSoftware is a Seb4Vision software built to do the following:
 
 * Capture Video Data
 * Prepare the Data for processing
 * Capture Snapshots
 * Provide an interface for Fill and Key
 * Connect to the VizMachine and give output there.

The Software is build on C/C++ and the Cuda Runtime.

Software Dependencies:
 1. [Cuda Toolkit - (11.2 - 11.6)](https://developer.nvidia.com/cuda-11-6-2-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local)   - [install instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) 
 2. [Nvidia TensorRT 8](https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.5.3/local_repos/nv-tensorrt-local-repo-ubuntu2004-8.5.3-cuda-11.8_1.0-1_amd64.deb) - [install instructions](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
 3. [OpenCV 4.5.2 with CudaSupport](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7)
 4. [cuDNN 8.9.](https://developer.nvidia.com/downloads/compute/cudnn/secure/8.9.1/local_installers/11.8/cudnn-local-repo-ubuntu2004-8.9.1.23_1.0-1_amd64.deb) - [install instructions](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
 5. Decklink 
 6. C++ Standard 9
 7. Ubuntu 20.04 (Focal)
 8. Eclipse (CDT) 
 9. G++ and GCC

***Please make sure that you install the Dependencies in this order***
1. Cuda
2. cuDNN
3. TensorRT
4. OpenCV

Hardware Dependencies:
1. Decklink Card
2. Nvidia Cuda Enabled GPU (Compute >=5)

***Building Process***
1. Make sure you have all the dependencies above working on your environment
2. Clone the repository using
```
git clone https://github.com/juriev101/CudaChromaUbuntu.git 
cd CudaChromaUbuntu
```
3. open up eclipse and import the project into your workspace *make sure you check the "copy project to workspace checkbox."
4. Navigate to the project->Properties->C/C++ build (expand)-> Settings-> NVCC linker-> libraries
5. Add all the neccessary libraries from opencv, cuda and tensorrt
6. Do the same with the include files
7. Congratulations you are ready to run!!!! 


*If there are any problems you face following these steps outlined above, <br /> please feel free to open up a git issue and we'll address it*

