FLAGS= -DDEBUG
clean:
	rm -f *.o nbody
nbody:
	nvcc -o nbody nbody.cu -lm
debug:
	nvcc $(FLAGS) -o nbody nbody.cu -lm
