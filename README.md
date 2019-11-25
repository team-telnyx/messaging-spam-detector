# For use

The train module will be used to train the initial data, Currently I'm just using plain csv as the machine's base data
Eventually a `.pickle` file will be used 

To run use main.py and specify which csv to be used, `combined_sms_mix.csv` is already trained and can be used for now!

A prediction is already set in place within main.py that uses a `.csv` file, the second predict module is also available
in predict.py which uses a `.pickle` file. Using a `.pickle` file is faster and recommended for use in terms of large files,
but the main advantage of it would be never displaying any real information in regards to the contents of a sms message. 