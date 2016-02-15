PYTHON_VERSION = 3.4
BOOST_PYTHON_LIB = boost_python3
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
THRUST_INCLUDE = /home/wrwt/Libraries/
NUMPY_INCLUDE = /usr/local/lib/python3.4/dist-packages/numpy/core/include/
BOOST_NUMPY_INCLUDE = /home/wrwt/Libraries/Boost.NumPy
BOOST_NUMPY_LIB_DIR = /home/wrwt/Libraries/Boost.NumPy/lib
PYTHON_LIB_DIR = /usr/lib/python$(PYTHON_VERSION)/config-3.4m-x86_64-linux-gnu
BOOST_INC = /home/wrwt/Libraries/boost_1_59_0/
BOOST_LIB_DIR = /home/wrwt/Libraries/boost_1_59_0/stage/lib/
#DEBUG = -DDEBUG
COMPILER = nvcc -std=c++11 $(DEBUG)
CFLAGS = -g

all: entcalc.so 

entcalc.so: wrapper.o 
	$(COMPILER) $(CFLAGS) -shared wrapper.o -L$(BOOST_LIB_DIR) -l$(BOOST_PYTHON_LIB) -L$(PYTHON_LIB_DIR) -lpython$(PYTHON_VERSION) -L$(BOOST_NUMPY_LIB_DIR) -lboost_numpy -larmadillo -lcublas -o entcalc.so

wrapper.o: wrapper.cu entropy_calculator.cu ndarray.h
	$(COMPILER) $(CFLAGS) -DPYTHON -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -I$(BOOST_NUMPY_INCLUDE) -I$(THRUST_INCLUDE) -I$(NUMPY_INCLUDE) --compiler-options="-fPIC" -c wrapper.cu -o wrapper.o

clean:
	rm -f : *.o *.so a.out main
