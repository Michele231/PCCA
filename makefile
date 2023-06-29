# makefile
# Michele Martinazzo

path_to_eigen=./lib/eigen-3.4.0/
path_to_spectra=./lib/spectra-master/include

CC = g++
CFLAGS = -Ofast -std=c++11 -I${path_to_eigen} -I${path_to_spectra}
LDFLAGS = -std=c++11

TARGET = cic

install: $(TARGET)

$(TARGET): CIC.cpp
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(TARGET)	
