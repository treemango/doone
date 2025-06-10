CXXFLAGS = 
CXX = g++ -std=c++20

TARGET = doone
SRC = src/*.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)

