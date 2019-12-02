# Makefile

CC = g++
CFLAGS = -O
INCPATH += `pkg-config --cflags opencv`
LIBS += `pkg-config --libs opencv`

EXEDIR = ./bin
OBJDIR = ./obj
SRCDIR = ./src

TARGET1 = test_simple_convnet

OBJ1 = $(OBJDIR)/test_simple_convnet.o $(OBJDIR)/simple_convnet.o $(OBJDIR)/conv2d.o $(OBJDIR)/maxpool2d.o $(OBJDIR)/relu.o $(OBJDIR)/linear.o $(OBJDIR)/softmax.o $(OBJDIR)/utils.o

all: $(TARGET1)

$(TARGET1): $(OBJ1)
	$(CC) $(LIBS) -o $(TARGET1) $^

$(OBJDIR)/test_simple_convnet.o: $(SRCDIR)/test_simple_convnet.cpp
	$(CC) $(CFLAGS) $(INCPATH) -c $< -o $@

$(OBJDIR)/simple_convnet.o: $(SRCDIR)/simple_convnet.cpp
	$(CC) $(CFLAGS) $(INCPATH) -c $< -o $@

$(OBJDIR)/conv2d.o: $(SRCDIR)/conv2d.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/maxpool2d.o: $(SRCDIR)/maxpool2d.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/relu.o: $(SRCDIR)/relu.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/linear.o: $(SRCDIR)/linear.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/softmax.o: $(SRCDIR)/softmax.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/utils.o: $(SRCDIR)/utils.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET1) $(OBJDIR)/*.o
