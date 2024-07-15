# Wave NN and parameter optimizer 


### V1

first iteration of the complete neural network and parameter optimization is in the v1 folder. It will not give correct results as the model has not been trained on a actual training set yet, however the code does work. However however, the code is also very very inefficient (n^2 or greater code) and thus does not scale well with the large data sets it will be solving and optimizing. The code can also be better orginized with calsses being properly set up and called in other programs.

### V2

working on v2 right now. I'm going to get rid of all the concatenating and the multiple arrays and just initialize a correctly sized array and populate it like I should have done at the start but was lazy. This should help massivly with the ranking times as that process currently takes around 10 hours for a 83k torque curves data set. Also going to actually make a seperate program for the wave_nn class so that the training and optomization programs can be cleaned up. There's more small tasks that will be done that I can't be bothered to mention. Also hopefully I can get the actual training set done for v2 but that might be a v3 thing. stupid wave  
