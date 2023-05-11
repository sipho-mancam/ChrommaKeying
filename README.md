# CudaChromaUbuntu

The CudaChromaSoftware is a Seb4Vision software built to do the following:
 
 * Capture Video Data
 * Prepare the Data for processing
 * Capture Snapshots
 * Provide an interface for Fill and Key
 * Connect to the VizMachine and give output there.

The Software is build on C/C++ and the Cuda Runtime.

Dependencies:
 1. [Cuda Toolkit - (11.2 - 11.6)]() 
 2. [Nvidia TensorRT 8] ()
 3. [OpenCV 4.5.2 with CudaSupport] ()
 4. [cuDNN 8.9.] ()
 5. Decklink 
 6. C++ Standard 9

***Building Process***
1. Make sure you have all the dependencies above working on your environment
2. Clone the repository using <br /> <code>git clone https://github.com/juriev101/CudaChromaUbuntu.git </code>
```
language bash
cd CudaChromaUbuntu
```
3. open up eclipse and import the project into your workspace *make sure you check the "copy project to workspace checkbox."
4. Navigate to the project->Properties->C/C++ build (expand)-> Settings-> NVCC linker-> libraries
5. Add all the neccessary libraries from opencv, cuda and tensorrt
6. 
