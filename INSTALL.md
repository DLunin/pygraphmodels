You should install a modified version of numpy (https://github.com/DLunin/numpy) to be able to create DGM with number of nodes >= 33
Also you need a C++ compiler and boost C++ library with boost.python

To compile C++ files:
  1. edit Makefiles (you will have to change libraries/interpreter paths)
  2. run `make`
  
