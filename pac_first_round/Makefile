
CXX=icpx

gpp_obj = main.o

CXXFLAGS = -Ofast -std=c++17   -qopenmp -xhost -fsycl -g  -fsycl-targets=nvptx64-nvidia-cuda
LDFLAGS = 

all: main 
EXEC= main

main: $(gpp_obj)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(gpp_obj) -o $(EXEC).exe -g

%.o: %.cu 
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@ -g 


clean:
	rm -f *.o $(EXEC).exe
