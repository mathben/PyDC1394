all: video1394.so

video1394.so: video1394.c
	cc video1394.c -I/usr/include/python2.7/ -I/usr/include/dc1394 -ldc1394 -lpython2.7 --shared -fPIC -o video1394.so -I/usr/lib/python2.7/site-packages/numpy/core/include/

video1394.c : video1394.pyx
	cython video1394.pyx

clean :
	rm video1394.c
	rm video1394.so
