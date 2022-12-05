
bench: exp_approx.cc
	g++ exp_approx.cc -O3 -march=native -mfma -o bench

run: bench
	./bench

clean:
	rm -f ./bench
