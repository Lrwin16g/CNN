# Makefile

CC = g++
CFLAGS = -O
#INCPATH += `pkg-config --cflags opencv`
#LIBS += `pkg-config --libs opencv`

EXEDIR = ./bin
OBJDIR = ./obj
SRCDIR = ./src

TARGET1 = train_convnet
TARGET2 = train_deepnet

OBJ1 = $(OBJDIR)/train_convnet.o $(OBJDIR)/simple_convnet.o $(OBJDIR)/conv2d.o \
	   $(OBJDIR)/maxpool2d.o $(OBJDIR)/relu.o $(OBJDIR)/linear.o $(OBJDIR)/softmax.o \
	   $(OBJDIR)/adam.o $(OBJDIR)/utils.o

OBJ2 = $(OBJDIR)/train_deepnet.o $(OBJDIR)/deep_convnet.o $(OBJDIR)/conv2d.o \
	   $(OBJDIR)/maxpool2d.o $(OBJDIR)/relu.o $(OBJDIR)/linear.o $(OBJDIR)/softmax.o \
	   $(OBJDIR)/dropout.o $(OBJDIR)/adam.o $(OBJDIR)/utils.o

all: $(TARGET1) $(TARGET2)

$(TARGET1): $(OBJ1)
	$(CC) $(CFLAGS) $(LIBS) -o $(TARGET1) $^

$(TARGET2): $(OBJ2)
	$(CC) $(CFLAGS) $(LIBS) -o $(TARGET2) $^

$(OBJDIR)/train_convnet.o: $(SRCDIR)/train_convnet.cpp
	$(CC) $(CFLAGS) $(INCPATH) -c $< -o $@

$(OBJDIR)/train_deepnet.o: $(SRCDIR)/train_deepnet.cpp
	$(CC) $(CFLAGS) $(INCPATH) -c $< -o $@

$(OBJDIR)/simple_convnet.o: $(SRCDIR)/simple_convnet.cpp
	$(CC) $(CFLAGS) $(INCPATH) -c $< -o $@

$(OBJDIR)/deep_convnet.o: $(SRCDIR)/deep_convnet.cpp
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

$(OBJDIR)/dropout.o: $(SRCDIR)/dropout.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/sgd.o: $(SRCDIR)/sgd.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/adam.o: $(SRCDIR)/adam.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/sigmoid.o: $(SRCDIR)/sigmoid.cpp
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR)/utils.o: $(SRCDIR)/utils.cpp
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET1) $(TARGET2) $(OBJDIR)/*.o
