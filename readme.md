computing energy
---
https://github.com/sunpho84/ising2/blob/3fcf39cacf9bbe198e10f4dc4429489e3f3173be/ising.cpp#L25-L40

gnuplot 
---

Open the program (after main)l
https://github.com/sunpho84/ising2/blob/badff1a75cebbd88a37eb5920fcfe5f940e1d2a5/ising.cpp#L34-L38

update the plot after conf is updated
https://github.com/sunpho84/ising2/blob/badff1a75cebbd88a37eb5920fcfe5f940e1d2a5/ising.cpp#L102-L109

close before the program ends
https://github.com/sunpho84/ising2/blob/badff1a75cebbd88a37eb5920fcfe5f940e1d2a5/ising.cpp#L115-L119


timings
---

Define measurement and time difference
https://github.com/sunpho84/ising2/blob/c30de05d38626f877cc70bb3ef1f6bc1f5614c90/ising.cpp#L10-L24

Taking time
https://github.com/sunpho84/ising2/blob/c30de05d38626f877cc70bb3ef1f6bc1f5614c90/ising.cpp#L90

Measuring compute energy
https://github.com/sunpho84/ising2/blob/c30de05d38626f877cc70bb3ef1f6bc1f5614c90/ising.cpp#L112-L115

Printing
https://github.com/sunpho84/ising2/blob/c30de05d38626f877cc70bb3ef1f6bc1f5614c90/ising.cpp#L160-L161

it can be needed to change `high_resolution_clock` into `steady_clock`


Thread parallelization
----------------------

```
g++ -o ising ising.cpp -fopenmp -O3 -Wall
```

launching to change the number of threads

```
OMP_NUM_THREADS=4 ./ising 40
```


Assignement Lect 5
-------------------

1) description of the system
2) description of the algorithm
3) plot of the simulation in the cold/hot phase
4) scaling of the time with the volume in the range L = 8 - 128
5) parallelization of the energy calculation: verify the reproducibility (same result is pbtained every time the code is run)
6) scaling of the time with the number of threads in two regimes: small and large volume
7) point out the different behavior in the two regimes


OMP_verbosity
-------------
```
export OMP_DISPLAY_AFFINITY=true
export OMP_DISPLAY_ENV=true
OMP_NUM_THREADS=4 OMP_PROC_BIND=true ./ising 
OMP_NUM_THREADS=4 OMP_PROC_BIND=true OMP_PLACES=1,2,3,4 ./ising
```

Scaling Lect 6
--------------
Show the strong scaling behavior of your code `T*N` as a function of `N` where `T` is the time and `N` is the number of threads, under the regime of small/large volume.
Alternatively, show the weak scaling of your code `T` as function of `N`, starting from a small/large volume at `N=1`.

workstation modeset
-------------------
scl enable devtoolset-11 bash

Scaling Lect 9
--------------

Parallelizing the algorithm
https://github.com/sunpho84/ising2/blob/a05af48f8239c9a067b54f64574affdab304533e/ising.cpp#L140-L145

Parallelizing the random number generator
https://github.com/sunpho84/ising2/blob/a05af48f8239c9a067b54f64574affdab304533e/ising.cpp#L121-L124

Using the local random generator
https://github.com/sunpho84/ising2/blob/a05af48f8239c9a067b54f64574affdab304533e/ising.cpp#L156
