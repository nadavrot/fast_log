
all: exp_approx log_approx log_accurate exp_accurate

exp_approx: exp_approx.cc util.h
	g++ exp_approx.cc -O3 -g -Wall -march=native -mfma -o exp_approx

log_approx: log_approx.cc util.h
	g++ log_approx.cc -O3 -g -Wall -march=native -mfma -o log_approx

log_accurate: log_accurate.cc util.h
	g++ log_accurate.cc -O3 -g -Wall -march=native -mfma -o log_accurate

exp_accurate: exp_accurate.cc util.h
	g++ exp_accurate.cc -O3 -g -Wall -march=native -mfma -o exp_accurate

clean:
	rm -f ./exp_approx ./log_approx ./log_accurate ./exp_accurate
