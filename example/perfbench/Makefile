.PHONY: all clean

GraphDotPath ?= $(shell findlink $PWD/../..)

all:


protein-time-to-solution:
		python3 protein-time-to-solution.py -active=2$(printf ",%d" $(for i in $(seq 63); do echo 2; done))

molecule-cookie-cutter:
		python3 molecule-cookie-cutter.py -formula='bicyclobutane' -repeat=32
