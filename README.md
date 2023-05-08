# PHYS-5070-Final-Project

This is my final project for PHYS 5070 SP23. My goal was to write a program that could calculate orbital transfer trajectories between any two bodies in a solar system given a variety of initial conditions. I had varied success with this project, mostly getting stuck dealing with the solver I used which proved to be very problematic.

I ended up making a little writeup in `FinalProject.ipynb`. It contains some of the steps that I took to solve the problem and my thinking as I worked through the project. I suggest referring to that if you want a description of the code. Other .ipynb notebooks contain some of the larger codes that I did not want to clutter that notebook up with.

`Tests.ipynb` : This notebook has some tests that I used to verify that my free space system was valid.

`FunctionScripts.ipynb` : This notebook writes the functions for me in a way such that the integrator can understand them

'five_object_solar_system.py' : This file is an example of the output from `FunctionScripts.ipynb`

`Mars_1.0_0.0_0.25.csv` : This is an example of a file where I saved the time, positions, and momentums for Mars over one path. I only included 1/25th of the values to keep the size down. Each shooting trial requires ~8mb of hard drive space for this method. 

I have also included my powerpoint slides.
