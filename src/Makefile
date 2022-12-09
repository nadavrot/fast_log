
all: exp_approx log_approx

exp_approx: exp_approx.cc util.h
	g++ exp_approx.cc -O3 -march=native -mfma -o exp_approx

log_approx: log_approx.cc util.h
	g++ log_approx.cc -O3 -march=native -mfma -o log_approx

clean:
	rm -f ./exp_approx ./log_approx
