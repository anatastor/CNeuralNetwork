
CC ?= gcc

PROG = example

LIB = neuralnet.a

OBJ = neuralnet.o

LDFLAGS = -lm


all: $(PROG) $(LIB)


$(PROG): $(PROG).c $(OBJ)
	$(CC) -o $@ $^ $(LDFLAGS)



.o: .c .h
	$(CC) -c $@ $^


$(LIB): $(PROG)
	ar rcs $@ $(OBJ)



.PHONY:
	all
	clean

clean:
	rm $(OBJ) $(LIB)
