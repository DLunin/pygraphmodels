target: structure

structure: structure.cpp mdarray.h
	g++ structure.cpp -std=c++14 -o structure 

test: test.cpp mdarray.h
	g++ -DDEBUG test.cpp -std=c++14 -o test -I/home/wrwt/Libraries/boost_1_60_0/

clean:
	rm -rf *.o structure test
