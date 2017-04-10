how to run:
1) run jupyter notebook project_2.ipynb
2) run py.test test.py

test.py can be run first before project_2.ipynp if trappist_1.compile() is run first.
plots and animations (maybe) are in the jupyter notebook.


good information - 
https://www.nasa.gov/press-release/nasa-telescope-reveals-largest-batch-of-earth-size-habitable-zone-planets-around

'trappist_1.py' holds main body of code.

'project_2.ipynb' plots trappist-1 orbits and has visual python animation (maybe) of trappist-1 system.

'npy' folder holds the following arrays:

N_t = 1e6 + 1
N_b = 7
dim = 2

name,	description,		dimention
T	time			N_t 
R	position		( N_b , N_t , dim )
V	velocity		( N_b , N_t , dim )
P	momentum		( N_b , N_t , dim )
KE	kinetic energy		( N_t , N_b )
UG	potential energy	( N_t , N_b )
E 	total energy		N_t
E_err	err in total energy	N_t

vis	visible planets		( N_b , N_t )
vis_av	average visible planets	N_t				
