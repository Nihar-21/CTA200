CC=gcc
DEPS=average.h
OBJ=main.o	average.o
CFLAGS=-std=c99
%.o: %.c $(DEPS)
	$(CC)	$(CFLAGS)	-c	-o	$@	$<

main: $(OBJ)
	$(CC)	-o	$@	$^	$(CFLAG)
