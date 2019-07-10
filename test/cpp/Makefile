CPP=$(wildcard catch_*.cpp)
TEST=$(CPP:.cpp=.test)
INC=-I. -I../../graphdot/cpp

.PHONY: clean

default: $(TEST)

clean:
	rm -f *.bin
	rm -f *.lammpstrj
	rm -f *.o

#catch_%.test : catch_%.gxx.bin catch_%.clg.bin
catch_%.test : catch_%.gxx.bin
	

catch_%.gxx.o: catch_%.cpp
	g++ -std=c++14 $(INC) -g -Og $< -c -o $@

catch_%.gxx.bin: catch_%.gxx.o catch.o
	g++ -g -Og $< catch.o -o $@
	./$@

catch_%.clg.o: catch_%.cpp
	clang++ -std=c++14 -Wno-c++1z-extensions $(INC) -g -Og $< -c -o $@

catch_%.clg.bin: catch_%.clg.o catch.o
	clang++ -g -Og $< catch.o -o $@
	./$@

catch.o: catch.cpp catch.hpp
	g++ -std=c++14 -g -Og -I. catch.cpp -c -o catch.o
