FLAGS= -DDEBUG
LIBS= -lm
clean:
	rm -f *.o nbody
nbody:
	nvcc -o nbody nbody.cu $(LIBS)
debug:
	nvcc $(FLAGS) -o nbody nbody.cu $(LIBS)
