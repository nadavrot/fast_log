
bench: bench.cc
	g++ bench.cc -O3 -march=native -mfma -o bench

run: bench
	./bench

clean:
	rm -f ./bench
