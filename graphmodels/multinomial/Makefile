PYTHON_VERSION = 2.7
PYTHON_VERSION_MAJOR = 2

BOOST_PYTHON_LIB = boost_python-$(PYTHON_VERSION_MAJOR)

PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
NUMPY_INCLUDE = /usr/local/lib/python$(PYTHON_VERSION)/dist-packages/numpy/core/include/
PYTHON_LIB_DIR = /usr/lib/python$(PYTHON_VERSION)/config-$(PYTHON_VERSION)m-x86_64-linux-gnu

BOOST_INC = /home/wrwt/Libraries/boost_1_60_0/
BOOST_LIB_DIR = /home/wrwt/Libraries/boost_1_60_0/stage/lib/
COMPILER = g++ -std=c++11 $(DEBUG)
CFLAGS = -g

multinomial_cpp.so: wrapper.o 
	$(COMPILER) $(CFLAGS) -shared -fPIC $^ -L$(BOOST_LIB_DIR) -l$(BOOST_PYTHON_LIB) -L$(PYTHON_LIB_DIR) -lpython$(PYTHON_VERSION) -o $@

wrapper.o: Makefile wrapper.cpp ndarray.h 
	$(COMPILER) $(CFLAGS) -DPYTHON$(PYTHON_VERSION_MAJOR) -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -I$(NUMPY_INCLUDE) -fPIC -c wrapper.cpp -o $@

clean:
	rm -f : *.o *.so a.out main
