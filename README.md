# The Magi System aka Wave NN and Parameter Optimizer

## New Version Details  
Wave NN v2 is done. Yay. Here's what's new:
 - The biggest improvement is a change in how the torque tensor is built from a concatenation approach to the pre-initializing a tensor and populating it. This change in approach reduced optimizer times for a 1.63 million case DOE table from around 11 hours to 25 minutes. Specifically, calculating ranking parameters and final rank for all the torque curves went from 8 hours to under 5 seconds :).
   
 - The model class and convergence function are now in their own program, this cleans up the training loop and helps with readability and redundancy in the program.
   
 - The optimization half of the program is now split into two programs. One that solves the DOE table and another that ranks and sorts the torque curves. This allows you to change how you want to rank the torque curves without having to solve the whole DOE table every time
   
 - the get_smoothness function is no longer straight up wrong
   
 - Other little house keeping things to make the program faster/easier to read. See v2 details for specifics.

## About This Project
  Wave NN is a neural network approach to optimizing engine package parameters to produce the "best" torque curve created by Michael Salas for use on Longhorn Racing, UT Austin's Formula SAE team. Traditionally the UT Formula SAE team has used Ricardo WAVE with some MATLAB scripting for finding optimized engine package parameters. And while this method works it is very difficult to test both a wide range of parameters and have a fine step size for them too. This is due to the fact that Ricardo WAVE is somewhere in the region of N^2 - N^3 code and thus doesn't scale well at all. For reference, our team used Ricardo WAVE to create 448 torque curves with a rpm step size of 500, this took 2 days across 7 computers. As to why we had to do this on 7 separate computers instead of one powerful one, WAVE is able to solve in a parallel manner and use all the cores you can give so it is generally very fast, the setup for each time step, however is done on one core and it is the setup that does not scale.
 
  Wave NN seeks to and succeeds in fixing these issues. By creating a large enough data set we are able to create a neural network that can predict a torque value based on the parameters it was trained on, such as intake and exhaust parameters and rpm. We can then use this model to solve full factorial tables for engine package parameters of interest and later rank and sort them to obtain the engine package parameters that create the best torque curve for our use case. In its current state, wave NN is around a 1,000,000x speed up versus Ricardo WAVE and the code scales at a N code versus the N^2 - N^3 of Ricardo WAVE.
 
  There are of course limitations to wave NN. It can only predict one variable per model, more output variables means more models. Training the model takes quite a bit of data, our team is anticipating having to train our model on around 228,000 cases which ironically need to be solved in Ricardo WAVE. We need to use the wave to kill the wave. And of course due to the fact that wave NN is trained on a full factorial DOE table, adding more input variables very quickly becomes computationally inefficient to do so, limiting the range of variables it's practical to optimize for.
 
  Now why is this project's actual name The Magi System? Well our entire engine package is named after Evangelion characters so why not continue the theme. I will, however, always refer to The Magi System as wave NN in documentation because I can't take seeing "The Magi System" written everywhere unironically. I already made that mistake calling our engine intake Asuka. I'm not doing it again.

## v2 Details
As mentioned in the new version details, v2 is a major jump in efficiency and ease of use. Here's a list of the features/good to know info for using wave NN:
 - The model class and converge criteria are in wavenn.py that is where you can change the model shape and convergence criteria values for training
 - training loop.py is where the model training is. replace the .csv file with whatever you want to train the model on
 - make sure you replace the .csv in wavenn.py so that correct normalization parameters are used during inference
 - get_avg_torque takes a upper and lower rpm bound to average over
 - if you train a model that takes more or less than 5 parameters the optimizer code won't work because a decent bit of the array indexing is still hard coded, I'll fix it at some point
 - the final raking value has weights assigned to each ranking parameter which can be changed based on what you want to prioritize

The code also has a lot of comments in case I forgot to mention anything here.

## Past Version(s)
### v1 details
first iteration of the complete neural network and parameter optimization is in the v1 folder. It will not give correct results as the model has not been trained on a actual training set yet, however the code does work. However however, the code is also very very inefficient (n^2 or greater code) and thus does not scale well with the large data sets it will be solving and optimizing. To put how inefficient this code is in perspective, in order to create one solved and ranked DOE table containing around 163,000,000 elements the program will create two other arrays, one with 97,000,000 elements and the other with 147,000,000 elements. AND IT WILL REPEATEDLY CALL 2 OF THESE ARRAYS AT ONCE TO CONCATENATE THEM. The code can also be better organized with classes being properly set up and called in other programs.
