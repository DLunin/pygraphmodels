target: structure

structure_debug: structure.cpp mdarray.h
	g++ -DDEBUG -g -O0 structure.cpp -std=c++14 -o structure_debug

structure: structure.cpp mdarray.h
	g++ -O3 structure.cpp -std=c++14 -o structure 

test: test.cpp mdarray.h
	g++ -DDEBUG test.cpp -std=c++14 -o test -I/home/wrwt/Libraries/boost_1_60_0/

clean:
	rm -rf *.o structure test structure_debug
