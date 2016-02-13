PYTHON_VERSION = 3.4
BOOST_PYTHON_LIB = boost_python3
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
NUMPY_INCLUDE = /usr/local/lib/python3.4/dist-packages/numpy/core/include/
BOOST_NUMPY_INCLUDE = /home/wrwt/Libraries/Boost.NumPy
BOOST_NUMPY_LIB_DIR = /home/wrwt/Libraries/Boost.NumPy/lib
PYTHON_LIB_DIR = /usr/lib/python$(PYTHON_VERSION)/config-3.4m-x86_64-linux-gnu
BOOST_INC = /home/wrwt/Libraries/boost_1_59_0/
BOOST_LIB_DIR = /home/wrwt/Libraries/boost_1_59_0/stage/lib/
COMPILER = g++ -std=c++11 $(DEBUG)
CFLAGS = -g

multinomial_cpp.so: wrapper.o 
	$(COMPILER) $(CFLAGS) -shared $^ -L$(BOOST_LIB_DIR) -l$(BOOST_PYTHON_LIB) -L$(PYTHON_LIB_DIR) -lpython$(PYTHON_VERSION) -L$(BOOST_NUMPY_LIB_DIR) -lboost_numpy -o $@

wrapper.o: wrapper.cpp ndarray.h
	$(COMPILER) $(CFLAGS) -DPYTHON -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -I$(BOOST_NUMPY_INCLUDE) -I$(NUMPY_INCLUDE) -fPIC -c wrapper.cpp -o $@

clean:
	rm -f : *.o *.so a.out main
