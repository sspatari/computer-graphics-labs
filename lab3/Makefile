NAME = image-processing

LINK.o = $(LINK.cc)
LDLIBS ?= -lcudart -lcuda
NVCCFLAGS ?= -arch compute_50 -Xcompiler -Wall,-Wextra,-fPIC

$(NAME): $(NAME).o lodepng.o

%.o: %.cu
	nvcc $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(NAME).o $(NAME) lodepng.o
